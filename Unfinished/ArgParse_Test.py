# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:18:07 2024

@author: CH242985
"""


import argparse

def main(*args):
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Just convert the args to a string-string dict so each model handles its own parsing.
    _, unknown_args = parser.parse_known_args()
    args = dict(zip([k[2:] for k in unknown_args[:-1:2]],unknown_args[1::2])) # from stackoverflow
    print(args) #these are all strings!
    # Run_Model(args)
    
    a = int( args['a'])
    b = int( args['b'])
    c = int( args['c'])
    
    simulate(a,b,c)
    
    
    
def simulate(a,b,c):
    print('adding!', a+b+c)
    
    

if __name__ == '__main__':
    main()