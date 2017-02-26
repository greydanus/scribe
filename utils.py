import numpy as np
import math
import random
import os
import cPickle as pickle
import xml.etree.ElementTree as ET

from utils import *

class DataLoader():
    def __init__(self, args, logger, limit = 500):
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        # self.tsteps = args.tsteps
        self.data_scale = args.data_scale # scale data down by this factor
        # self.ascii_steps = args.tsteps/args.tsteps_per_ascii
        self.logger = logger
        self.limit = limit # removes large noisy gaps in the data

        data_file = os.path.join(self.data_dir, "strokes_training_data.cpkl")
        stroke_dir = self.data_dir + "/lineStrokes"
        ascii_dir = self.data_dir + "/ascii"

        if not (os.path.exists(data_file)) :
            self.logger.write("\tcreating training data cpkl file from raw source")
            self.preprocess(stroke_dir, ascii_dir, data_file)

        self.load_preprocessed(data_file)
        self.reset_batch_pointer()

    def preprocess(self, stroke_dir, ascii_dir, data_file):
        # create data file from raw xml files from iam handwriting source.
        self.logger.write("\tparsing dataset...")
        
        # build the list of xml files
        filelist = []
        # Set the directory you want to start from
        rootDir = stroke_dir
        for dirName, subdirList, fileList in os.walk(rootDir):
            for fname in fileList:
                filelist.append(dirName+"/"+fname)

        # function to read each individual xml file
        def getStrokes(filename):
            tree = ET.parse(filename)
            root = tree.getroot()

            result = []

            x_offset = 1e20
            y_offset = 1e20
            y_height = 0
            for i in range(1, 4):
                x_offset = min(x_offset, float(root[0][i].attrib['x']))
                y_offset = min(y_offset, float(root[0][i].attrib['y']))
                y_height = max(y_height, float(root[0][i].attrib['y']))
            y_height -= y_offset
            x_offset -= 100
            y_offset -= 100

            for stroke in root[1].findall('Stroke'):
                points = []
                for point in stroke.findall('Point'):
                    points.append([float(point.attrib['x'])-x_offset,float(point.attrib['y'])-y_offset])
                result.append(points)
            return result
        
        # function to read each individual xml file
        def getAscii(filename, line_number):
            with open(filename, "r") as f:
                s = f.read()
            s = s[s.find("CSR"):]
            if len(s.split("\n")) > line_number+2:
                s = s.split("\n")[line_number+2]
                s = s.strip()
                return s
            else:
                return ""
                
        # converts a list of arrays into a 2d numpy int16 array
        def convert_stroke_to_array(stroke):
            n_point = 0
            for i in range(len(stroke)):
                n_point += len(stroke[i])
            stroke_data = np.zeros((n_point, 3), dtype=np.int16)

            prev_x = 0
            prev_y = 0
            counter = 0

            for j in range(len(stroke)):
                for k in range(len(stroke[j])):
                    stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
                    stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
                    prev_x = int(stroke[j][k][0])
                    prev_y = int(stroke[j][k][1])
                    stroke_data[counter, 2] = 0
                    if (k == (len(stroke[j])-1)): # end of stroke
                        stroke_data[counter, 2] = 1
                    counter += 1
            return stroke_data

        # pad strokes to equal length
        def strokes_postprocess(strokes, longest_stroke):
            padlen = longest_stroke + 1
            numstrokes = len(strokes)
            stroke_data = np.zeros((numstrokes, padlen, 3), dtype=np.int16)
            for n in range(numstrokes):
                len_current_stroke = len(strokes[n])
                stroke_data[n][:len_current_stroke] = strokes[n]
            return np.array(stroke_data)

        # pad ascii strings to equal length
        def ascii_postprocess(ascii, longest_ascii):
            padlen = longest_ascii + 1
            ascii_data = []
            for s in ascii:
                ascii_data.append(s.ljust(padlen, '_'))
            return ascii_data

        # build stroke database of every xml file inside iam database
        strokes = []
        asciis = []
        alphabet = set('_')
        longest_stroke = 0
        longest_ascii = 0
        for i in range(len(filelist)):
            if (filelist[i][-3:] == 'xml'):
                stroke_file = filelist[i]
#                 print 'processing '+stroke_file
                stroke = convert_stroke_to_array(getStrokes(stroke_file))
                
                ascii_file = stroke_file.replace("lineStrokes","ascii")[:-7] + ".txt"
                line_number = stroke_file[-6:-4]
                line_number = int(line_number) - 1
                ascii = getAscii(ascii_file, line_number)
                alphabet = alphabet.union(set(ascii))
                if len(stroke) > longest_stroke:
                    longest_stroke = len(stroke)
                if len(ascii) > longest_ascii:
                    longest_ascii = len(ascii)
                strokes.append(stroke)
                asciis.append(ascii)

        print("Longest stroke: {}".format(longest_stroke))
        print("Longest ascii: {}".format(longest_ascii))
        assert(len(strokes)==len(asciis)), "There should be a 1:1 correspondence between stroke data and ascii labels."
        strokes = strokes_postprocess(strokes, longest_stroke)
        asciis = ascii_postprocess(asciis, longest_ascii)
        stroke_length = strokes[0].shape[0]
        ascii_length = len(asciis[0])
        alphabet_string = "".join(sorted(alphabet))
        metadata = {
            'stroke_length': stroke_length,
            'ascii_length': ascii_length,
            'alphabet': alphabet_string
            }
        print("Metadata: {}".format(metadata))
        f = open(data_file,"wb")
        pickle.dump([metadata,strokes,asciis], f, protocol=2)
        f.close()
        self.logger.write("\tfinished parsing dataset. saved {} lines".format(len(strokes)))


    def load_preprocessed(self, data_file):
        f = open(data_file,"rb")
        [metadata, self.raw_stroke_data, self.raw_ascii_data] = pickle.load(f)
        f.close()
        self.alphabet = metadata['alphabet']
        self.stroke_length = metadata['stroke_length']
        self.ascii_length = metadata['ascii_length']
        self.tsteps = self.stroke_length
        self.ascii_steps = self.ascii_length

        # goes thru the list, and only keeps the text entries that have more than tsteps points
        self.stroke_data = []
        self.ascii_data = []
        self.valid_stroke_data = []
        self.valid_ascii_data = []
        counter = 0
        num_training_chars = 0
        num_valid_chars = 0

        # every 1 in 20 (5%) will be used for validation data
        cur_data_counter = 0
        for i in range(len(self.raw_stroke_data)):
            data = self.raw_stroke_data[i]
            # print("COMPARE {} and {}".format(len(data), (self.tsteps+2)))
            if len(data) == self.tsteps:
            # if len(data) > (self.tsteps+2):
                # removes large gaps from the data
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data,dtype=np.float32)
                data[:,0:2] /= self.data_scale
                cur_data_counter = cur_data_counter + 1
                cur_ascii = self.raw_ascii_data[i]
                if cur_data_counter % 20 == 0:
                    self.valid_stroke_data.append(data)
                    self.valid_ascii_data.append(cur_ascii)
                    num_valid_chars += len(cur_ascii.translate(None, '_'))
                else:
                    self.stroke_data.append(data)
                    self.ascii_data.append(cur_ascii)
                    num_training_chars += len(cur_ascii.translate(None, '_'))

        # minus 1, since we want the ydata to be a shifted version of x data
        self.num_batches = int(len(self.stroke_data) / self.batch_size)
        self.logger.write("\tloaded dataset:")
        self.logger.write("\t\t{} stroke length".format(self.stroke_length))
        self.logger.write("\t\t{} ascii length".format(self.ascii_length))
        self.logger.write("\t\t{} alphabet length".format(len(self.alphabet)))
        self.logger.write("\t\t{} train characters over {} sequences".format(num_training_chars, len(self.stroke_data)))
        self.logger.write("\t\t{} valid characters over {} sequences".format(num_valid_chars, len(self.valid_stroke_data)))
        self.logger.write("\t\t{} training batches".format(self.num_batches))

    def validation_data(self):
        # returns validation data
        x_batch = []
        y_batch = []
        ascii_list = []
        for i in range(self.batch_size):
            valid_ix = i%len(self.valid_stroke_data)
            data = self.valid_stroke_data[valid_ix]
            data = np.vstack((data, [0, 0, 0]))
            x_batch.append(np.copy(data[:self.tsteps]))
            y_batch.append(np.copy(data[1:self.tsteps+1]))
            ascii_list.append(self.valid_ascii_data[valid_ix])
        one_hots = [to_one_hot(s, self.ascii_steps, self.alphabet) for s in ascii_list]
        return x_batch, y_batch, ascii_list, one_hots

    def next_batch(self):
        # returns a randomized, tsteps-sized portion of the training data
        x_batch = []
        y_batch = []
        ascii_list = []
        for i in xrange(self.batch_size):
            data = self.stroke_data[self.idx_perm[self.pointer]]
            data = np.vstack((data, [0, 0, 0]))
            x_batch.append(np.copy(data[:self.tsteps]))
            y_batch.append(np.copy(data[1:self.tsteps+1]))
            ascii_list.append(self.ascii_data[self.idx_perm[self.pointer]])
            self.tick_batch_pointer()
        one_hots = [to_one_hot(s, self.ascii_steps, self.alphabet) for s in ascii_list]
        return x_batch, y_batch, ascii_list, one_hots

    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.stroke_data)):
            self.reset_batch_pointer()
    def reset_batch_pointer(self):
        self.idx_perm = np.random.permutation(len(self.stroke_data))
        self.pointer = 0

# utility function for converting input ascii characters into vectors the network can understand.
# index position 0 means "unknown"
def to_one_hot(s, ascii_steps, alphabet):
    steplimit=3e3; s = s[:3e3] if len(s) > 3e3 else s # clip super-long strings
    seq = [alphabet.find(char) + 1 for char in s]
    if len(seq) >= ascii_steps:
        seq = seq[:ascii_steps]
    else:
        seq = seq + [0]*(ascii_steps - len(seq))
    one_hot = np.zeros((ascii_steps,len(alphabet)+1))
    one_hot[np.arange(ascii_steps),seq] = 1
    return one_hot

# abstraction for logging
class Logger():
    def __init__(self, args):
        self.logf = '{}train_scribe.txt'.format(args.log_dir) if args.train else '{}sample_scribe.txt'.format(args.log_dir)
        with open(self.logf, 'w') as f: f.write("Scribe: Realistic Handriting in Tensorflow\n     by Sam Greydanus\n\n\n")

    def write(self, s, print_it=True):
        if print_it:
            print s
        with open(self.logf, 'a') as f:
            f.write(s + "\n")
