-- mixture density network criterion
-- Jiakai Zhang, Elman Mansimov
require('cunn')

local MDNCriterion, parent = torch.class('nn.MDNCriterion', 'nn.Criterion')

function MDNCriterion:__init(nGaussians)
   parent.__init(self)
   self.sizeAverage = true
   self.output = 0
   self.gradInput = torch.Tensor()

   self.nGaussians = nGaussians
   -- init transfer function
   self.transf =  nn.ParallelTable()
   self.transf:add(nn.Identity())   -- mu
   self.transf:add(nn.Exp())        -- sigma
   self.transf:add(nn.SoftMax())    -- alpha
   --self.transf:cuda()
   self.mu    = torch.Tensor()
   self.sigma = torch.Tensor()
   self.alpha = torch.Tensor()
   self.pin   = torch.Tensor() -- prior possibility
   --
   self.input = {}
   self.transfInput = {}
   --
   self.delta = 1e-3
   self.sigma_min = 1e-1
end

-- input is a table of mu, sigma and alpha
function MDNCriterion:updateOutput(input, target)
   assert(target:dim() == 2, "target should be 2-D tensor")
   assert(target:size(2) == 1, "target should have only 1 dim features")
   local nSamples = target:size(1)

   self.pin:resize(nSamples, self.nGaussians)
   self.input = input
   self.transfInput = self.transf:forward(self.input)

   self.mu    = self.transfInput[1]
   self.sigma = self.transfInput[2]
   self.alpha = self.transfInput[3]

   local ds = torch.pow(self.mu - target:repeatTensor(1, self.nGaussians), 2) * 0.5
   ds:cdiv(torch.pow(self.sigma, 2))
   local phi = torch.exp(-ds) / math.sqrt(2 * math.pi)
   phi:cdiv(self.sigma):cmul(self.alpha)
   self.pin:copy(phi)
   local loss = -torch.log(self.pin:sum(2))

   self.output = loss:sum()
   if self.sizeAverage == true then
      self.output = self.output / nSamples
   end

   --print('loss=' .. self.output)
   return self.output
end

function MDNCriterion:updateGradInput(input, target)
   -- input is a table
   self.gradInput = {}
   for i = 1, #input do
      self.gradInput[i] = torch.Tensor()
      self.gradInput[i]:resizeAs(input[i])
   end

   local pin_sum = self.pin:sum(2)
   local nSamples = target:size(1)
   self.pin:cdiv(pin_sum:repeatTensor(1, self.nGaussians))

   --local mu    = self.transfInput[1]
   --local sigma = self.transfInput[2]
   --local alpha = self.transfInput[3]

   local d_mu    = self.gradInput[1]
   local d_sigma = self.gradInput[2]
   local d_alpha = self.gradInput[3]

   local dist = self.mu - target:repeatTensor(1, self.nGaussians)
   d_mu:copy(torch.cmul(self.pin, torch.cdiv(dist, torch.pow(self.sigma, 2))) / nSamples)
   d_sigma:copy(torch.cmul(self.pin, (-torch.cdiv(torch.pow(dist, 2), torch.pow(self.sigma, 2)) + 1)) / nSamples)
   d_alpha:copy((self.alpha - self.pin) / nSamples)

   return self.gradInput
end
