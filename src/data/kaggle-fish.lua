local argcheck = require 'argcheck'
local tnt   = require 'torchnet'
local utils  = require 'torchnet-vision.datasets.utils'
local utils2 = require 'src.data.utils'
local lsplit = string.split

local dataset = {}

local function splitDataset(dataset, classes, percent, fnametrain, fnametest)
   local counts = {}
   for i=1, #classes do
      counts[i] = 0
   end 
   utils2.iterOverDataset(dataset, dataset:size(), {
      onSample = function(sample)
         counts[sample.target] = counts[sample.target] + 1
      end
   })

   local percents = {}
   for i=1, #classes do
      percents[i] = math.floor(counts[i] * percent * 1.0 / 100)
   end 

   local ftrain = assert(io.open(fnametrain, 'w'))
   local ftest  = assert(io.open(fnametest, 'w'))

   local counts = {}
   for i=1, #classes do
      counts[i] = 0
   end 
   utils2.iterOverDataset(dataset, dataset:size(), {
      onSample = function(sample)
         counts[sample.target] = counts[sample.target] + 1
         if counts[sample.target] < percents[sample.target] then
            ftrain:write(sample.line)
            ftrain:write("\n")
         else
            ftest:write(sample.line)
            ftest:write("\n")
         end
      end
   })
   ftrain:close()
   ftest:close()
end

dataset.load = argcheck{
   {name='dirname', type='string', default='data'},
   call =
      function(dirname)
         local dir_raw = paths.concat(dirname, 'raw')
         local dir_train = paths.concat(dir_raw, 'train', 'train')
         local path_filename = paths.concat(dir_train, 'filename.txt')

         --local dir_interim = paths.concat(dirname, 'interim')

         local dir_processed = paths.concat(dirname, 'processed')
         local path_fname_train = paths.concat(dir_processed, 'fnametrain.txt')
         local path_fname_val  = paths.concat(dir_processed, 'fnameval.txt')

         local classes = utils.findClasses(dir_train)
         local class2target = {}
         for i, class in ipairs(classes) do
            class2target[class] = i
         end
         
         if not paths.filep(path_filename) then
            utils.findFilenames(dir_train, classes)
         end

         local load = function(line)
            local splt = lsplit(line, '/')
            local sample = {
               line = line,
               path = paths.concat(dir_train, line),
               class = splt[1],
               target = class2target[splt[1]]
            }
            return sample
         end

         if not paths.filep(path_fname_train) and
            not paths.filep(path_fname_val) then

            local dataset = tnt.ListDataset{
               filename = path_filename,
               load = load
            }
            dataset = dataset:shuffle()

            splitDataset(dataset, classes, 85, path_fname_train, path_fname_val)
         end

         local trainset = tnt.ListDataset{
            filename = path_fname_train,
            load = load
         }
         local valset = tnt.ListDataset{
            filename = path_fname_val,
            load = load
         }
         return trainset, valset, classes, class2target
      end
}

dataset.loadTrainset = argcheck{
   {name='dirname', type='string', default='data'},
   call =
      function(dirname)
         local dir_raw = paths.concat(dirname, 'raw')
         local dir_train = paths.concat(dir_raw, 'train', 'train')
         local path_filename = paths.concat(dir_train, 'filename.txt')

         --local dir_interim = paths.concat(dirname, 'interim')

         local dir_processed = paths.concat(dirname, 'processed')

         local classes = utils.findClasses(dir_train)
         local class2target = {}
         for i, class in ipairs(classes) do
            class2target[class] = i
         end
         
         if not paths.filep(path_filename) then
            utils.findFilenames(dir_train, classes)
         end

         local load = function(line)
            local splt = lsplit(line, '/')
            local sample = {
               line = line,
               path = paths.concat(dir_train, line),
               class = splt[1],
               label = splt[1],
               target = class2target[splt[1]]
            }
            return sample
         end

         local trainset = tnt.ListDataset{
            filename = path_filename,
            load = load
         }

         return trainset, classes, class2target
      end
}

--dataset.load()

return dataset
