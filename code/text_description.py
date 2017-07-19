import numpy as np
from abc import ABCMeta, abstractmethod


class Description:

    def __init__(self):
        self.init_states = []
        self.events = []
        self.global_props = []
        self.digits = []

    def __str__(self):
        str_list = []
        # Initial state
        str_list.append(' , and '.join(str(x) for x in self.init_states))
        str_list.append('.')
        str_list.append('then ,')
        str_list.append(' . meanwhile , '.join(str(x) for x in self.events))
        str_list.append('.')
        return ' '.join(str_list)


class DigitInitialState:

    def __init__(self, digit):
        self.qualifiers = []
        self.digit = digit
        self.actions = []

    def __str__(self):
        ret_list = ['the']
        if len(self.qualifiers) > 0:
            ret_list.append(' , '.join([str(x) for x in self.qualifiers]))
        ret_list.append(str(self.digit))
        ret_list.append(str(self.actions[0]))
        if len(self.actions) > 1:
            ret_list.append('as it')
            for i in range(1, len(self.actions)-1):
                ret_list.append('%s and' % str(self.actions[i]))
            ret_list.append(str(self.actions[-1]))

        return ' '.join(ret_list)


class DigitEvent:

    def __init__(self, digit):
        self.qualifiers = []
        self.digit = digit
        self.actions = []

    def __str__(self):
        ret_list = ['the']
        if len(self.qualifiers) > 0:
            ret_list.append(' , '.join([str(x) for x in self.qualifiers]))
        ret_list.append(str(self.digit))
        # Append each action with "then" in between each one
        actions_str = ' , then '.join([str(x) for x in self.actions])
        ret_list.append(actions_str)

        return ' '.join(ret_list)


class Entity:

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Digit(Entity):

    def __init__(self, id, label):
        self.id = id
        self.label = label

    def __str__(self):
        return str(self.label)
        # return '[Digit id=%d, label=%d]' % (self.id, self.label)


class Wall(Entity):
    NORTH = 'north wall'
    SOUTH = 'south wall'
    EAST = 'east wall'
    WEST = 'west wall'

    def __init__(self, direction):
        self.direction = direction

    def __str__(self):
        return self.direction


class Adjective:
    BIG = 'big'
    SMALL = 'small'
    GROW = 'growing'
    SHRINK = 'shrinking'

    def __init__(self, adjective):
        self.adverbs = []
        self.adjective = adjective


    def __str__(self):
        return ' '.join(self.adverbs + [self.adjective])


class Verb:
    MOVE = 'moves'
    HIT = 'hits'
    ROTATE = 'rotates'
    GROWS = 'grows'
    SHRINKS = 'shrinks'

    def __init__(self, verb, direct_object=None):
        self.verb = verb
        self.adverbs = []
        self.direct_object = direct_object

    def __str__(self):
        ret_list = [self.verb]
        if self.direct_object:
            ret_list.append('the')
            ret_list.append(str(self.direct_object))
        ret_list += [str(x) for x in self.adverbs]
        return ' '.join(ret_list)


class Adverb:
    CW = 'clockwise'
    CCW = 'counterclockwise'

    NORTH = 'north'
    NW = 'northwest'
    WEST = 'west'
    SW = 'southwest'
    SOUTH = 'south'
    SE = 'southeast'
    EAST = 'east'
    NE = 'northeast'

    RAPIDLY = 'rapidly'
    SLOWLY = 'slowly'

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass


class GlobalProperty:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

class FlashingProperty(GlobalProperty):

    def __init__(self, rate):
        self.rate = rate

    def __str__(self):
        return '[FlashingProperty rate=%d]' % self.rate


def point_to_cardinal_dir(x, y):
    '''
    Return the adjective string corresponding to the octant that the given point is in
    :param x: Horizontal position
    :param y: Vertical position, with positive y going northward
    :return:
    '''
    angle = (np.arctan2(y, x) * 180 / np.pi) % 360
    if angle > 22.5 and angle <= 67.5:
        return Adverb.NE
    elif angle > 67.5 and angle <= 112.5:
        return Adverb.NORTH
    elif angle > 112.5 and angle <= 157.5:
        return Adverb.NW
    elif angle > 157.5 and angle <= 202.5:
        return Adverb.WEST
    elif angle > 202.5 and angle <= 247.5:
        return Adverb.SW
    elif angle > 247.5 and angle <= 292.5:
        return Adverb.SOUTH
    elif angle > 292.5 and angle <= 337.5:
        return Adverb.SE
    else:
        return Adverb.EAST


def count_to_enumeration(count):
    '''
    Return the adjective string corresponding to the count (e.g. 1st, 2nd, 3rd)
    :param count:
    :return:
    '''
    if count >= 10 and count <= 19:
        return '%dth' % count
    elif (count % 10) == 1:
        return '%dst' % count
    elif (count % 10) == 2:
        return '%dnd' % count
    elif (count % 10) == 3:
        return '%drd' % count
    else:
        return '%dth' % count