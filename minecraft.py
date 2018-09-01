from minecraft_supportings import *

class MineCraft(pyglet.window.Window,gym.Env):

    def __init__(self, args=None, obs_size=84, obs_type='gray', episode_length_limit=100):

        self.args = args
        self.obs_size = obs_size
        if self.obs_size not in [84]:
            raise Exception('Check outside agent to make you want this.')
        self.obs_type = obs_type
        if self.obs_type not in ['gray']:
            raise Exception('Check outside agent to make you want this.')

        self.obs_type_to_num_channel = {
            'gray': 1,
            'rgb': 3,
        }
        self.obs_type_to_gl_type = {
            'gray':GL_LUMINANCE,
            'rgb':GL_RGB,
        }
        self.obs_type_to_pil_mode = {
            'gray':"L",
            'rgb':"RGB",
        }

        '''init for MineCraft'''
        super(MineCraft, self).__init__(width=self.obs_size, height=self.obs_size, caption='Pyglet', resizable=True, visible=False)
        # Whether or not the window exclusively captures the mouse.
        self.exclusive = False
        # A list of blocks the player can place. Hit num keys to cycle.
        self.inventory = [BRICK, GRASS, SAND]
        # Convenience list of num keys.
        self.num_keys = [
            key._1, key._2, key._3, key._4, key._5,
            key._6, key._7, key._8, key._9, key._0]

        self.step_interval = 0.1
        self.buffer = (GLubyte*(self.obs_type_to_num_channel[self.obs_type]*self.width*self.height))(0)

        '''init for gym env'''
        self.episode_length_limit = episode_length_limit
        if self.episode_length_limit<0:
            raise Exception('episode_length_limit < 0 means episode is only terminated by done signal, please check this.')
        self.action_to_key_map = {
            0: key.A,
            1: key.W,
            2: key.S,
            3: key.D,
            4: key.Q,
            5: key.E,
            6: key.T,
            7: key.F,
            8: key.G,
            9: key.H,
        }
        self.key_map_to_action = {}
        for key_i in self.action_to_key_map.keys():
            self.key_map_to_action[self.action_to_key_map[key_i]]=key_i

        self.action_space = spaces.Discrete(len(self.action_to_key_map.keys()))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, self.obs_type_to_num_channel[self.obs_type]),dtype=np.uint8)
        # Instance of the model that handles the world.
        self.model = Model()

    def seed(self,seed):
        print("# WARNING: Deterministic game")

    def set_exclusive_mouse(self, exclusive):
        """ If `exclusive` is True, the game will capture the mouse, if False
        the game will ignore the mouse.

        """
        super(Window, self).set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    def get_sight_vector(self):
        """ Returns the current line of sight vector indicating the direction
        the player is looking.

        """
        x, y = self.rotation
        # y ranges from -90 to 90, or -pi/2 to pi/2, so m ranges from 0 to 1 and
        # is 1 when looking ahead parallel to the ground and 0 when looking
        # straight up or down.
        m = math.cos(math.radians(y))
        # dy ranges from -1 to 1 and is -1 when looking straight down and 1 when
        # looking straight up.
        dy = math.sin(math.radians(y))
        dx = math.cos(math.radians(x - 90)) * m
        dz = math.sin(math.radians(x - 90)) * m
        return (dx, dy, dz)

    def get_motion_vector(self):
        """ Returns the current motion vector indicating the velocity of the
        player.

        Returns
        -------
        vector : tuple of len 3
            Tuple containing the velocity in x, y, and z respectively.

        """
        if any(self.strafe):
            x, y = self.rotation
            strafe = math.degrees(math.atan2(*self.strafe))
            y_angle = math.radians(y)
            x_angle = math.radians(x + strafe)
            if self.flying:
                m = math.cos(y_angle)
                dy = math.sin(y_angle)
                if self.strafe[1]:
                    # Moving left or right.
                    dy = 0.0
                    m = 1
                if self.strafe[0] > 0:
                    # Moving backwards.
                    dy *= -1
                # When you are flying up or down, you have less left and right
                # motion.
                dx = math.cos(x_angle) * m
                dz = math.sin(x_angle) * m
            else:
                dy = 0.0
                dx = math.cos(x_angle)
                dz = math.sin(x_angle)
        else:
            dy = 0.0
            dx = 0.0
            dz = 0.0
        return (dx, dy, dz)

    def update(self, dt):
        """ This method is scheduled to be called repeatedly by the pyglet
        clock.

        Parameters
        ----------
        dt : float
            The change in time since the last call.

        """
        self.model.process_queue()
        sector = sectorize(self.position)
        if sector != self.sector:
            self.model.change_sectors(self.sector, sector)
            if self.sector is None:
                self.model.process_entire_queue()
            self.sector = sector
        m = 8
        dt = min(dt, 0.2)
        for _ in range(m):
            self._update(dt / m)

    def _update(self, dt):
        """ Private implementation of the `update()` method. This is where most
        of the motion logic lives, along with gravity and collision detection.

        Parameters
        ----------
        dt : float
            The change in time since the last call.

        """
        # walking
        speed = FLYING_SPEED if self.flying else WALKING_SPEED
        d = dt * speed # distance covered this tick.
        dx, dy, dz = self.get_motion_vector()
        # New position in space, before accounting for gravity.
        dx, dy, dz = dx * d, dy * d, dz * d
        # gravity
        if not self.flying:
            # Update your vertical speed: if you are falling, speed up until you
            # hit terminal velocity; if you are jumping, slow down until you
            # start falling.
            self.dy -= dt * GRAVITY
            self.dy = max(self.dy, -TERMINAL_VELOCITY)
            dy += self.dy * dt
        # collisions
        x, y, z = self.position
        x, y, z = self.collide((x + dx, y + dy, z + dz), PLAYER_HEIGHT)
        self.position = (x, y, z)

    def collide(self, position, height):
        """ Checks to see if the player at the given `position` and `height`
        is colliding with any blocks in the world.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position to check for collisions at.
        height : int or float
            The height of the player.

        Returns
        -------
        position : tuple of len 3
            The new position of the player taking into account collisions.

        """
        # How much overlap with a dimension of a surrounding block you need to
        # have to count as a collision. If 0, touching terrain at all counts as
        # a collision. If .49, you sink into the ground, as if walking through
        # tall grass. If >= .5, you'll fall through the ground.
        pad = 0.25
        p = list(position)
        np = normalize(position)
        for face in FACES:  # check all surrounding blocks
            for i in range(3):  # check each dimension independently
                if not face[i]:
                    continue
                # How much overlap you have with this dimension.
                d = (p[i] - np[i]) * face[i]
                if d < pad:
                    continue
                for dy in range(height):  # check each height
                    op = list(np)
                    op[1] -= dy
                    op[i] += face[i]
                    if tuple(op) not in self.model.world:
                        continue
                    p[i] -= (d - pad) * face[i]
                    if face == (0, -1, 0) or face == (0, 1, 0):
                        # You are colliding with the ground or ceiling, so stop
                        # falling / rising.
                        self.dy = 0
                    break
        return tuple(p)

    def on_mouse_press(self, x, y, button, modifiers):
        """ Called when a mouse button is pressed. See pyglet docs for button
        amd modifier mappings.

        Parameters
        ----------
        x, y : int
            The coordinates of the mouse click. Always center of the screen if
            the mouse is captured.
        button : int
            Number representing mouse button that was clicked. 1 = left button,
            4 = right button.
        modifiers : int
            Number representing any modifying keys that were pressed when the
            mouse button was clicked.

        """
        print('on_mouse_press')
        if self.exclusive:
            vector = self.get_sight_vector()
            block, previous = self.model.hit_test(self.position, vector)
            if (button == mouse.RIGHT) or \
                    ((button == mouse.LEFT) and (modifiers & key.MOD_CTRL)):
                # ON OSX, control + left click = right click.
                if previous:
                    self.model.add_block(previous, self.block)
            elif button == pyglet.window.mouse.LEFT and block:
                texture = self.model.world[block]
                if texture != STONE:
                    self.model.remove_block(block)
        else:
            self.set_exclusive_mouse(True)

    def on_mouse_motion(self, x, y, dx, dy):
        """ Called when the player moves the mouse.

        Parameters
        ----------
        x, y : int
            The coordinates of the mouse click. Always center of the screen if
            the mouse is captured.
        dx, dy : float
            The movement of the mouse.

        """
        if self.exclusive:
            m = 0.15
            x, y = self.rotation
            x, y = x + dx * m, y + dy * m
            y = max(-90, min(90, y))
            self.rotation = (x, y)

    def move_view(self, dx, dy):
        """ Called when the player moves the view.
        """
        m = 1.5
        x, y = self.rotation
        x, y = x + dx * m, y + dy * m
        y = max(-90, min(90, y))
        self.rotation = (x, y)

    def on_key_press(self, symbol, modifiers):
        """ Called when the player presses a key. See pyglet docs for key
        mappings.

        Parameters
        ----------
        symbol : int
            Number representing the key that was pressed.
        modifiers : int
            Number representing any modifying keys that were pressed.

        """
        self.strafe[0] = 0
        self.strafe[1] = 0
        if symbol == key.W:
            self.strafe[0] = -1
        elif symbol == key.S:
            self.strafe[0] = 1
        elif symbol == key.A:
            self.strafe[1] = -1
        elif symbol == key.D:
            self.strafe[1] = 1
        elif symbol == key.Q:
            vector = self.get_sight_vector()
            block, previous = self.model.hit_test(self.position, vector)
            if previous:
                self.model.add_block(previous, self.block)
        elif symbol == key.E:
            vector = self.get_sight_vector()
            block, previous = self.model.hit_test(self.position, vector)
            if block:
                texture = self.model.world[block]
                if texture != STONE:
                    self.model.remove_block(block)
        elif symbol == key.H:
            self.move_view(1,0)
        elif symbol == key.F:
            self.move_view(-1,0)
        elif symbol == key.T:
            self.move_view(0,1)
        elif symbol == key.G:
            self.move_view(0,-1)
        elif symbol == key.SPACE:
            if self.dy == 0:
                self.dy = JUMP_SPEED
        elif symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif symbol == key.TAB:
            self.flying = not self.flying
        elif symbol in self.num_keys:
            index = (symbol - self.num_keys[0]) % len(self.inventory)
            self.block = self.inventory[index]

    def on_key_release(self, symbol, modifiers):
        """ Called when the player releases a key. See pyglet docs for key
        mappings.

        Parameters
        ----------
        symbol : int
            Number representing the key that was pressed.
        modifiers : int
            Number representing any modifying keys that were pressed.

        """
        print('on_key_release')
        if symbol == key.W:
            self.strafe[0] += 1
        elif symbol == key.S:
            self.strafe[0] -= 1
        elif symbol == key.A:
            self.strafe[1] += 1
        elif symbol == key.D:
            self.strafe[1] -= 1

    def on_resize(self, width, height):
        """ Called when the window is resized to a new `width` and `height`.

        """

        # # label
        # self.label.y = height - 10

        # reticle
        # if self.reticle:
        #     self.reticle.delete()
        # x, y = self.width // 2, self.height // 2
        # n = 10
        # self.reticle = pyglet.graphics.vertex_list(4,
        #     ('v2i', (x - n, y, x + n, y, x, y - n, x, y + n))
        # )
        pass

    def set_2d(self):
        """ Configure OpenGL to draw in 2d.

        """
        width, height = self.get_size()
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def set_3d(self):
        """ Configure OpenGL to draw in 3d.

        """
        width, height = self.get_size()
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65.0, width / float(height), 0.1, 60.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        x, y = self.rotation
        glRotatef(x, 0, 1, 0)
        glRotatef(-y, math.cos(math.radians(x)), 0, math.sin(math.radians(x)))
        x, y, z = self.position
        glTranslatef(-x, -y, -z)

    def on_draw(self):
        """ Called by pyglet to draw the canvas.

        """
        self.clear()
        self.set_3d()
        glColor3d(1, 1, 1)
        self.model.batch.draw()
        self.draw_focused_block()
        self.set_2d()
        # self.draw_label()
        # self.draw_reticle()

    def draw_focused_block(self):
        """ Draw black edges around the block that is currently under the
        crosshairs.

        """
        vector = self.get_sight_vector()
        block = self.model.hit_test(self.position, vector)[0]
        if block:
            x, y, z = block
            vertex_data = cube_vertices(x, y, z, 0.51)
            glColor3d(0, 0, 0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            pyglet.graphics.draw(24, GL_QUADS, ('v3f/static', vertex_data))
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def draw_label(self):
        """ Draw the label in the top left of the screen.

        """
        x, y, z = self.position
        self.label.text = '%02d (%.2f, %.2f, %.2f) %d / %d' % (
            pyglet.clock.get_fps(), x, y, z,
            len(self.model._shown), len(self.model.world))
        self.label.draw()

    def draw_reticle(self):
        """ Draw the crosshairs in the center of the screen.

        """
        glColor3d(0, 0, 0)
        self.reticle.draw(GL_LINES)

    def get_obs(self):
        glReadPixels(0, 0, self.width, self.height, self.obs_type_to_gl_type[self.obs_type], GL_UNSIGNED_BYTE, self.buffer)
        obs = np.array(Image.frombytes(mode=self.obs_type_to_pil_mode[self.obs_type], size=(self.obs_size, self.obs_size), data=self.buffer).transpose(Image.FLIP_TOP_BOTTOM))
        if self.obs_type in ['gray']:
            obs=np.expand_dims(obs,2)
        return obs

    def reset_minecraft(self):
        """ Reset all settings in MineCraft.

        """

        # When flying gravity has no effect and speed is increased.
        self.flying = False

        # Strafing is moving lateral to the direction you are facing,
        # e.g. moving to the left or right while continuing to face forward.
        #
        # First element is -1 when moving forward, 1 when moving back, and 0
        # otherwise. The second element is -1 when moving left, 1 when moving
        # right, and 0 otherwise.
        self.strafe = [0, 0]

        # Current (x, y, z) position in the world, specified with floats. Note
        # that, perhaps unlike in math class, the y-axis is the vertical axis.
        self.position = (0, 0, 0)

        # First element is rotation of the player in the x-z plane (ground
        # plane) measured from the z-axis down. The second is the rotation
        # angle from the ground plane up. Rotation is in degrees.
        #
        # The vertical plane rotation ranges from -90 (looking straight down) to
        # 90 (looking straight up). The horizontal rotation range is unbounded.
        self.rotation = (0, -15)

        # Which sector the player is currently in.
        self.sector = None

        # Velocity in the y (upward) direction.
        self.dy = 0

        # The current block the user can place. Hit num keys to cycle.
        self.block = self.inventory[0]

    def reset(self):
        self.eposide_length = 0

        self.reset_minecraft()

        '''MineCraft has to step forward for one step to generate obs'''
        for i in range(1):
            self.update(self.step_interval)
            self.on_draw()
        obs = self.get_obs()

        if self.is_render:
            cv2.imshow('obs',obs)

        return obs

    def step(self, action):

        done = False
        self.eposide_length += 1

        '''step forward and get obs'''
        self.on_key_press(self.action_to_key_map[action],0)
        self.update(self.step_interval)
        self.on_draw()
        obs = self.get_obs()

        if self.is_render:
            cv2.imshow('obs',obs)

        '''generate reward'''
        reward = 0

        '''generate done'''
        if self.episode_length_limit > 0:
            if self.eposide_length >= self.episode_length_limit:
                done = True

        '''generate info'''
        info = None

        return obs, reward, done, info

    def set_render(self, is_render):
        self.is_render = is_render

def setup_fog():
    """ Configure the OpenGL fog properties.

    """
    # Enable fog. Fog "blends a fog color with each rasterized pixel fragment's
    # post-texturing color."
    glEnable(GL_FOG)
    # Set the fog color.
    glFogfv(GL_FOG_COLOR, (GLfloat * 4)(0.5, 0.69, 1.0, 1))
    # Say we have no preference between rendering speed and quality.
    glHint(GL_FOG_HINT, GL_DONT_CARE)
    # Specify the equation used to compute the blending factor.
    glFogi(GL_FOG_MODE, GL_LINEAR)
    # How close and far away fog starts and ends. The closer the start and end,
    # the denser the fog in the fog range.
    glFogf(GL_FOG_START, 20.0)
    glFogf(GL_FOG_END, 60.0)

def minecraft_global_setup():
    """ Basic OpenGL configuration.

    """
    # Set the color of "clear", i.e. the sky, in rgba.
    glClearColor(0.5, 0.69, 1.0, 1)
    # Enable culling (not rendering) of back-facing facets -- facets that aren't
    # visible to you.
    glEnable(GL_CULL_FACE)
    # Set the texture minification/magnification function to GL_NEAREST (nearest
    # in Manhattan distance) to the specified texture coordinates. GL_NEAREST
    # "is generally faster than GL_LINEAR, but it can produce textured images
    # with sharper edges because the transition between texture elements is not
    # as smooth."
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    setup_fog()

def main():

    minecraft_global_setup()

    env = MineCraft()
    env.set_render(True)

    obs = env.reset()
    action = env.key_map_to_action[cv2.waitKey(0)]

    while True:

        if action==ord('j'):
            cv2.destroyAllWindows()
            break

        obs, reward, done, info = env.step(action)
        # print(obs.shape)
        if done:
            env.reset()
        action = env.key_map_to_action[cv2.waitKey(0)]

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
