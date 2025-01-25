class ManualPolicy:
    """
    Leader uses wasd, Follower uses ijkl
    """
    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]

        self.show_obs = show_obs

        # action mappings for all agents are the same
        self.action_mapping = dict()
        if agent_id == 0:
            self.action_mapping['w'] = 0  # up
            self.action_mapping['s'] = 1  # down
            self.action_mapping['a'] = 2  # left
            self.action_mapping['d'] = 3  # right
        elif agent_id == 1:
            self.action_mapping = dict()
            self.action_mapping['i'] = 0  # up
            self.action_mapping['k'] = 1  # down
            self.action_mapping['j'] = 2  # left
            self.action_mapping['l'] = 3  # right
        else:
            assert False

    def __call__(self, observation, agent):
        # only trigger when we are the correct agent
        assert (
            agent == self.agent
        ), f"Manual Policy only applied to agent: {self.agent}, but got tag for {agent}."

        # set the default action
        key = ''
        while not key or key not in self.action_mapping:
            key = input(f'Agent {agent} input:\n')

        return self.action_mapping[key]

    @property
    def available_agents(self):
        return self.env.agent_name_mapping
