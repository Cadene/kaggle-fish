require 'image'
require 'os'
require 'optim'
ffi                  = require 'ffi'
local tnt            = require 'torchnet'
local vision         = require 'torchnet-vision'
local logtext        = require 'torchnet.log.view.text'
local logstatus      = require 'torchnet.log.view.status'
local transformimage = require 'torchnet-vision.image.transformimage'
local model          = require 'torchnet-vision.models.resnet'
local utils          = require 'src.data.utils'
local dataset        = require 'src.data.kaggle-fish'

local cmd = torch.CmdLine()
-- options to get acctop1=79.25 in 4 epoch
cmd:option('-seed', 1337, 'seed for cpu and gpu')
cmd:option('-usegpu', true, 'use gpu')
cmd:option('-bsize', 10, 'batch size')
cmd:option('-nepoch', 20, 'epoch number')
cmd:option('-nthread', 3, 'threads number for parallel iterator')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

config.idGPU = os.getenv('CUDA_VISIBLE_DEVICES') or -1
config.date  = os.date("%y_%m_%d_%X")

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(config.seed)

local path = '.'
local pathmodel   = path..'/models/resnet152.t7'
local pathdataset = path..'/data/processed'
local pathextract     = path..'/data/extract/'..config.date
local pathclasses      = pathdataset..'/classes.t7'
local pathclass2target = pathdataset..'/class2target.t7'
local pathconfig    = pathextract..'/config.t7'

local trainset, classes, class2target = dataset.loadTrainset()

require 'cudnn'
local net = model.load{
   filename = pathmodel,
   length   = 152
}
net:remove()
print(net)

local criterion = nn.CrossEntropyCriterion():float()

local function addTransforms(dataset, model)
   dataset = dataset:transform(function(sample)
      local img = image.load(sample.path, 3)
      sample.width = img:size(2)
      sample.height = img:size(3)
      sample.whratio = 1.0 * sample.width / sample.height
      sample.input  = tnt.transform.compose{
         vision.image.transformimage.scale(224),
         vision.image.transformimage.randomCrop(224),
         vision.image.transformimage.colorNormalize{
            mean = model.mean,
            std  = model.std
         }
      }(img)
      return sample
   end)
   return dataset
end
trainset = addTransforms(trainset, model)

local function getIterator(dataset)
   -- mode = {train,val,test}
   local iterator = tnt.ParallelDatasetIterator{
      nthread   = config.nthread,
      init      = function()
         require 'torchnet'
         require 'torchnet-vision'
      end,
      closure   = function(threadid)
         return dataset:batch(config.bsize)
      end,
      transform = function(sample)
         sample.target = torch.Tensor(sample.target):view(-1,1)
         return sample
      end
   }
   for i, v in pairs(iterator:exec('size')) do
      print(i, v)
   end
   return iterator
end
local trainiter = getIterator(trainset)

local meter = {
   timem = tnt.TimeMeter{unit = false},
}

local file
local engine = tnt.OptimEngine()
engine.mode = 'train'

local onStart = function(state)
   for _,m in pairs(meter) do m:reset() end

   engine.ibatch = 0
   engine.nbatch = state.iterator:execSingle("size")

   os.execute('mkdir -p '..pathextract)
   file = assert(io.open(pathextract..'/'..engine.mode..'set.csv', "w"))
   file:write('path;gttarget;gtclass;width;height;whratio')
   for i=1, 2048 do
      file:write(';feats'..i)
   end
   file:write("\n")
end

local onSample
if config.usegpu then
   require 'cutorch'
   cutorch.manualSeed(config.seed)
   require 'cunn'
   require 'cudnn'
   cudnn.convert(net, cudnn)
   net = net:cuda()
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  
end

local onForward = function(state)
   engine.ibatch = engine.ibatch + 1
   local output = state.network.output
   for i=1, output:size(1) do
      file:write(state.sample.path[i]);
      if engine.mode ~= 'test' then
         file:write(';'); file:write(state.sample.target[i][1])
         file:write(';'); file:write(state.sample.label[i])
         file:write(';'); file:write(state.sample.width[i])
         file:write(';'); file:write(state.sample.height[i])
         file:write(';'); file:write(state.sample.whratio[i])
      end
      for j=1, output:size(2) do
         file:write(';'); file:write(output[i][j])
      end
      file:write("\n")
   end
   xlua.progress(engine.ibatch, engine.nbatch)
end

local onEnd = function(state)
   print('End of extracting on '..engine.mode..'set')
   print('Took '..meter.timem:value())
   file:close()
end

engine.hooks.onStart   = onStart
engine.hooks.onSample  = onSample
engine.hooks.onForward = onForward
engine.hooks.onEnd     = onEnd

print('Extracting trainset ...')

engine:test{
   network   = net,
   iterator  = trainiter
}