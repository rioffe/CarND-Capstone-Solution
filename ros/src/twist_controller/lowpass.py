
class LowPassFilter(object):
    def __init__(self, tau, ts):
        self.a = 1. / (tau / ts + 1.)
        self.b = tau / ts / (tau / ts + 1.);

        self.last_val = 0.
        self.ready = False

    def get(self):
        return self.last_val

    def filt(self, val):
        if self.ready:
            val = self.a * val + self.b * self.last_val
        else:
            self.ready = True

        self.last_val = val
        return val

class LowPassFilter4(object):
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.sum = a + b + c + d

        self.l1 = 0.
        self.l2 = 0.
        self.l3 = 0.
        self.ready = False

    def get(self):
        return self.l3

    def filt(self, val):
        if self.ready:
            val = (self.a * val + self.b * self.l1 + self.c * self.l2 + self.d * self.l3)/self.sum
        else:
            self.ready = True

        self.l1 = self.l2
        self.l2 = self.l3
        self.l3 = val
        return val

    def reset(self):
        self.l1 = 0.
        self.l2 = 0.
        self.l3 = 0.
        self.ready = False


class LowPassFilter8(object):
    def __init__(self, a, b, c, d, e, f, g, h):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.h = h
        self.sum = a + b + c + d + e + f + g + h

        self.l1 = 0.
        self.l2 = 0.
        self.l3 = 0.
        self.l4 = 0.
        self.l5 = 0.
        self.l6 = 0.
        self.l7 = 0.
        self.ready = False

    def get(self):
        return self.l7

    def filt(self, val):
        if self.ready:
            val = (self.a * val + self.b * self.l1 + self.c * self.l2 + self.d * self.l3 + self.e * self.l4 + self.f * self.l5 + self.g * self.l6 + self.h * self.l7)/self.sum
        else:
            self.ready = True

        self.l1 = self.l2
        self.l2 = self.l3
        self.l3 = self.l4
        self.l4 = self.l5
        self.l5 = self.l6
        self.l6 = self.l7
        self.l7 = val
        return val

    def reset(self):
        self.l1 = 0.
        self.l2 = 0.
        self.l3 = 0.
        self.l4 = 0.
        self.l5 = 0.
        self.l6 = 0.
        self.l7 = 0.
        self.ready = False
