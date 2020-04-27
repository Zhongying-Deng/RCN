import os

class NIRVIS2(object):
    """casia nirvis 2.0"""
    def __init__(self, root):
        super(NIRVIS2, self).__init__()
        self.root = root
        self.splits = 10
        self.views = 2
        self.path_protocol = os.path.join(root, 'protocols')
        self.dev_vis_train_flist = 'vis_train_dev.txt'
        self.dev_nir_train_flist = 'nir_train_dev.txt'
        self.dev_probe_flist = 'nir_probe_dev.txt'
        self.dev_gallery_flist = 'vis_gallery_dev.txt'
        self.eval_probe_gallery_flists = []
        self.eval_vis_train_flists = []
        self.eval_nir_train_flists = []

    def get_eval_flists(self):
        # probe & gallery file lists for evaluation
        for i in xrange(1,self.splits+1):
            self.eval_probe_gallery_flists.append(['nir_probe_{}.txt'.format(i),'vis_gallery_{}.txt'.format(i)])
            self.eval_vis_train_flists.append('vis_train_{}'.format(i))
            self.eval_nir_train_flists.append('nir_train_{}'.format(i))


            



        