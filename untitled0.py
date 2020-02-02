# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:44:03 2018

@author: Muhammad Yahya
"""

class Example:
    name = "Example"

    @staticmethod
    def static():
        print ("%s static() called" % Example.name)

class Offspring1(Example):
    name = "Offspring1"

class Offspring2(Example):
    name = "Offspring2"

    @staticmethod
    def static():
        print ("%s static() called" % Offspring2.name)

Example.static() # prints Example
Offspring1.static() # prints Example
Offspring2.static() # prints Offspring2