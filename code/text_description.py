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
        self.location = None
        self.actions = []

    def __str__(self):
        ret_list = ['the']
        if len(self.qualifiers) > 0:
            ret_list.append(' , '.join([str(x) for x in self.qualifiers]))
        ret_list.append(str(self.digit))
        if self.location:
            ret_list.append('in the %s' % self.location)
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


class Location:

    CENTER = 'center region'
    NORTH = 'north region'
    NW = 'northwest region'
    WEST = 'west region'
    SW = 'southwest region'
    SOUTH = 'south region'
    SE = 'southeast region'
    EAST = 'east region'
    NE = 'northeast region'

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass


def grid_pos_to_location(x, y, video_size):
    horiz_grid_pos = int(np.floor(float(x) / video_size[0] * 3))
    vert_grid_pos = int(np.floor(float(y) / video_size[1] * 3))
    grid = [
        [Location.NW, Location.NORTH, Location.NE],
        [Location.WEST, Location.CENTER, Location.EAST],
        [Location.SW, Location.SOUTH, Location.SE],
    ]
    return grid[vert_grid_pos][horiz_grid_pos]


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
        self.qualifiers = []

    def __str__(self):
        ret_list = []
        if len(self.qualifiers) > 0:
            ret_list.append(' , '.join([str(x) for x in self.qualifiers]))
        ret_list.append(str(self.label))
        return ' '.join(ret_list)


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
    BLINK = 'blinking'
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'

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
    STAND_STILL = 'stands still'
    DO_NOTHING = 'does nothing else'
    OVERLAPS = 'overlaps'

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


def create_description_from_logger(logger,
                                   describe_location=True,
                                   describe_init_scale_speed=True,
                                   describe_reverse_scale_speed=True,
                                   describe_reverse_angle_speed=True,
                                   describe_hit_digit=True,
                                   describe_hit_wall=True,
                                   describe_overlap=True):
    '''
    Generate description from messages stored in the logger
    :param logger:
    :return:
    '''
    desc = Description()

    ### Initial state ###
    start_messages = []
    for message in logger.messages:
        if message['step'] != -1:
            break
        start_messages.append(message)

    digit_messages = filter(lambda x: x['type'] == 'digit', start_messages)
    digit_metas_list = [message['meta'] for message in digit_messages]
    digit_metas_list.sort(key=lambda x: x['id'])

    start_state_messages = filter(lambda x: x['type'] == 'start_state', start_messages)
    start_states_list = [message['meta'] for message in start_state_messages]
    start_states_list.sort(key=lambda x: x['digit_id'])

    update_params_messages = filter(lambda x: x['type'] == 'start_update_params', start_messages)
    update_params_list = [message['meta'] for message in update_params_messages]
    update_params_list.sort(key=lambda x: x['digit_id'])

    color_messages = filter(lambda x: x['type'] == 'digit_color', start_messages)
    color_list = [message['meta'] for message in color_messages]
    color_list.sort(key=lambda x: x['digit_id'])

    settings_message = filter(lambda x: x['type'] == 'settings', start_messages)[0]
    settings = settings_message['meta']

    # Populate total counts
    num_digits = len(digit_messages)
    total_label_counts = np.zeros(10)
    for meta in digit_metas_list:
        total_label_counts[meta['label']] += 1
    # Init seen count
    seen_label_counts = np.zeros(10)

    for i in range(num_digits):
        digit_info = digit_metas_list[i]
        state = start_states_list[i]
        update_params = update_params_list[i]

        # Create digit description
        digit_id, label = digit_info['id'], digit_info['label']
        digit = Digit(digit_id, label)

        # Enumerate if multiple of the same digit exist
        seen_label_counts[label] += 1
        if total_label_counts[label] > 1:
            adj = Adjective(count_to_enumeration(seen_label_counts[label]))
            digit.qualifiers.append(adj)
        desc.digits.append(digit)

        digit_init_state = DigitInitialState(digit)
        events = DigitEvent(digit)

        # Describe scale transform
        if describe_init_scale_speed:
            if update_params['scale_speed'] > 0:
                adj = Adjective(Adjective.GROW)
                digit_init_state.qualifiers.append(adj)
            elif update_params['scale_speed'] < 0:
                adj = Adjective(Adjective.SHRINK)
                digit_init_state.qualifiers.append(adj)

        # Describe flashing
        if settings['blink_rate'] > 1:
            adj = Adjective(Adjective.BLINK)
            digit_init_state.qualifiers.append(adj)

        # Describe translation
        x_speed = update_params['x_speed']
        y_speed = update_params['y_speed']
        if x_speed != 0 or y_speed != 0:
            verb = Verb(Verb.MOVE)
            # Direction
            verb.adverbs.append(point_to_cardinal_dir(x_speed, -y_speed))
            # Add verb
            digit_init_state.actions.append(verb)

        # Describe rotation
        angle_speed = update_params['angle_speed']
        if angle_speed != 0:
            verb = Verb(Verb.ROTATE)
            # Direction
            verb.adverbs.append(Adverb.CW if angle_speed > 0 else Adverb.CCW)
            # Add verb
            digit_init_state.actions.append(verb)

        # Describe color if supported color is used
        if len(color_list) > 0:
            color = color_list[i]['color']
            if np.array_equal(color, [255, 0, 0]):
                adj = Adjective(Adjective.RED)
                digit_init_state.qualifiers.append(adj)
            if np.array_equal(color, [0, 255, 0]):
                adj = Adjective(Adjective.GREEN)
                digit_init_state.qualifiers.append(adj)
            if np.array_equal(color, [0, 0, 255]):
                adj = Adjective(Adjective.BLUE)
                digit_init_state.qualifiers.append(adj)

        # Describe starting location
        if describe_location:
            video_size = settings['video_size']
            location = grid_pos_to_location(state['x'], state['y'], video_size)
            digit_init_state.location = location

        # If no translation or rotation occurs, describe as standing still
        if x_speed == 0 and y_speed == 0 and angle_speed == 0:
            verb = Verb(Verb.STAND_STILL)
            digit_init_state.actions.append(verb)

        # Finally, add to description
        desc.init_states.append(digit_init_state)
        desc.events.append(events)

    ### Events ###
    nonstart_messages = filter(lambda x: x['step'] != -1, logger.messages)
    overlap_step_map = {}
    for message in nonstart_messages:
        message_type, meta, step = message['type'], message['meta'], message['step']
        if message_type == 'reverse_scale_speed' and describe_reverse_scale_speed:
            digit_id = meta['digit_id']
            verb = Verb(Verb.SHRINKS if meta['new_direction'] < 0 else Verb.GROWS)
            desc.events[digit_id].actions.append(verb)
        elif message_type == 'reverse_angle_speed' and describe_reverse_angle_speed:
            digit_id = meta['digit_id']
            verb = Verb(Verb.ROTATE)
            verb.adverbs.append(Adverb.CW if meta['new_direction'] > 0 else Adverb.CCW)
            desc.events[digit_id].actions.append(verb)
        elif message_type == 'bounce_off_digit' and describe_hit_digit:
            # TODO: Also describe reverse interaction?
            digit_id = meta['digit_id_a']
            other_digit_id = meta['digit_id_b']
            verb = Verb(Verb.HIT)
            verb.direct_object = desc.digits[other_digit_id]
            desc.events[digit_id].actions.append(verb)
        elif message_type == 'bounce_off_wall' and describe_hit_wall:
            digit_id = meta['digit_id']
            wall_label = meta['wall_label']
            verb = Verb(Verb.HIT)
            if wall_label == 'north':
                wall_entity = Wall(Wall.NORTH)
            elif wall_label == 'south':
                wall_entity = Wall(Wall.SOUTH)
            elif wall_label == 'east':
                wall_entity = Wall(Wall.EAST)
            elif wall_label == 'west':
                wall_entity = Wall(Wall.WEST)
            verb.direct_object = wall_entity
            desc.events[digit_id].actions.append(verb)
        elif message_type == 'overlap' and describe_overlap:
            id_a = meta['digit_id_a']
            id_b = meta['digit_id_b']
            last_overlap_step = overlap_step_map.get((id_a, id_b), None)
            if last_overlap_step is None or last_overlap_step != step-1:
                verb = Verb(Verb.OVERLAPS)
                verb.direct_object = desc.digits[id_b]
                desc.events[id_a].actions.append(verb)
            overlap_step_map[(id_a, id_b)] = step


    # If any digit does nothing, add the do-nothing description
    for i in range(num_digits):
        if len(desc.events[i].actions) == 0:
            verb = Verb(Verb.DO_NOTHING)
            desc.events[i].actions.append(verb)

    return desc