import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class MPEVisualizer(object):
    def __init__(
        self,
        env,
        state_seq: list,
        reward_seq=None,
        env_name=None,
    ):
        self.env = env

        self.interval = 100
        self.state_seq = state_seq
        self.reward_seq = reward_seq

        self.env_name = env_name
        
        self.comm_active = not np.all(self.env.silent)
        print('Comm active? ', self.comm_active)
        
        self.init_render()

    def animate(
        self,
        save_fname: Optional[str] = None,
        view: bool = False,
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.state_seq),
            blit=False,
            interval=self.interval,
        )
        # Save the animation to a gif
        if save_fname is not None:
            ani.save(save_fname)

        # if view:
        #     plt.show(block=True)

    def init_render(self):
        from matplotlib.patches import Circle
        state = self.state_seq[0]
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        
        ax_lim = 2
        self.ax.set_xlim([-ax_lim, ax_lim])
        self.ax.set_ylim([-ax_lim, ax_lim])
        
        self.entity_artists = []
        self.labels = []
        for i in range(self.env.num_agents):
            if self.env_name == "MPE_simple_fire":
                # in FireEnv, draw agents as empty circles so they can be seen if they overlap
                c = Circle(
                    state.p_pos[i], state.rad[i], edgecolor=np.array(self.env.colour[i]) / 255, fill=False, facecolor='none',
                )
            elif self.env_name == "MPE_simple_transport":
                c = Circle(
                    state.p_pos[i], state.rad[i], edgecolor=np.array(self.env.colour[i]) / 255
                )
                x, y = c.center
                self.labels.append(self.ax.annotate(f"{state.capacity[i]}", (x+1.25*state.rad[i], y), color="black", ha="left", va="center", size=6))
            else:
                # otherwise default to filled circles
                c = Circle(
                    state.p_pos[i], state.rad[i], color=np.array(self.env.colour[i]) / 255
                )

            self.ax.add_patch(c)
            self.entity_artists.append(c)

        for j in range(self.env.num_landmarks):
            if self.env_name == "MPE_simple_fire":
                # in FireEnv, draw agents as empty circles so they can be seen if they overlap
                i = j + self.env.num_agents
                c = Circle(
                    state.p_pos[i], state.rad[i], edgecolor=np.array(self.env.colour[i]) / 255, fill=False, facecolor='none',
                )
            elif self.env_name == "MPE_simple_transport":
                i = j + self.env.num_agents
                c = Circle(
                    state.p_pos[i], state.rad[i], edgecolor=np.array(self.env.colour[i]) / 255, fill=False, facecolor='none',
                )
                x, y = c.center
                if j == 2:
                    self.labels.append(
                        self.ax.annotate(f"{state.site_quota}",
                        (x, y),
                        color="black", ha="left", va="center", size=8)
                    )
                else:
                    self.labels.append(None)
            else:
                # otherwise default to filled circles
                i = j + self.env.num_agents
                c = Circle(
                    state.p_pos[i], state.rad[i], color=np.array(self.env.colour[i]) / 255
                )
            self.ax.add_patch(c)
            self.entity_artists.append(c)
            
        self.step_counter = self.ax.text(-1.95, 1.95, f"Step: {state.step}", va="top")
        
        if self.comm_active:
            self.comm_idx = np.where(self.env.silent == 0)[0]
            print('comm idx', self.comm_idx)
            self.comm_artists = []
            i = 0
            for idx in self.comm_idx:
                
                letter = ALPHABET[np.argmax(state.c[idx])]
                a = self.ax.text(-1.95, -1.95 + i*0.17, f"{self.env.agents[idx]} sends {letter}")
                
                self.comm_artists.append(a)
                i += 1
            
    def update(self, frame):
        state = self.state_seq[frame]
        for i, c in enumerate(self.entity_artists):
            c.center = state.p_pos[i]

        for i in range(len(self.labels)):
            if self.labels[i]:
                x, y = self.entity_artists[i].center
                self.labels[i].set_x(x+1.25*state.rad[i])
                self.labels[i].set_y(y)
            if self.env_name == "MPE_simple_transport" and i == len(self.labels)-1:
                self.labels[i].set_text(f"{state.site_quota}")
            
        self.step_counter.set_text(f"Step: {state.step}")
        
        if self.comm_active:
            for i, a in enumerate(self.comm_artists):
                idx = self.comm_idx[i]
                letter = ALPHABET[np.argmax(state.c[idx])]
                a.set_text(f"{self.env.agents[idx]} sends {letter}")
        
