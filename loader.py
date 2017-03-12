import os

from processor import Clip

class DataLoader:

    def load_dataset(self, name):
        """Load all dataset recordings into a nested list."""
        clips = []
        for directory in sorted(os.listdir('{0}/'.format(name))):
            directory = '{0}/{1}'.format(name, directory)
            if os.path.isdir(directory) and os.path.basename(directory)[0:3].isdigit():
                print('Parsing ' + directory)
                category = []
                for clip in sorted(os.listdir(directory)):
                    if clip[-3:] == 'ogg':
                        category.append(Clip('{0}/{1}'.format(directory, clip)))
                clips.append(category)
        print('All {0} recordings loaded.'.format(name))
        return clips
