# coding=utf-8

from __init__ import *

class Tree_Ring:

    def __init__(self):
        self.this_class_arr = results_root + 'arr\\Tree_Ring\\'
        self.this_class_tif = results_root + 'tif\\Tree_Ring\\'
        self.this_class_png = results_root + 'png\\Tree_Ring\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass

    def run(self):

        pass

def main():
    Tree_Ring().run()
    pass


if __name__ == '__main__':

    main()