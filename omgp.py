#!/usr/bin/python

def ff(*xx):
    # xx is received as a tuple.
    print ("first arg:%d" % xx[0])
    print ("total args:%d" % len(xx))
    return xx

rr = ff(3, 4, 5)

print (rr)                        # (3,4,5)
