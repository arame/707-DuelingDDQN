from device import Device
class Config:
    n_games = 300
    lr = 0.0001
    gamma = 0.99
    eps_min = 0.01
    eps_decay = 0.999
    epsilon = 1.0
    max_mem = 5000
    repeat = 4
    batch_size = 32
    replace_target_cnt = 1000
    env_name = 'PongNoFrameskip-v4'
    path = 'models'
    load_checkpoint = False
    algo = 'DuelingDQNAgent'
    clip_rewards = False
    no_ops = 0
    fire_first = False
    chkpt_dir='checkpoint'
    Device.get()
    device = Device.device
    device_type = Device.type
    figure_file = 'plots/' + algo + '_' + env_name + '_lr' + str(lr) +'_'  + str(n_games) + 'games.png'


    @staticmethod
    def print_settings():
        print("\n"*10)
        print("*"*100)
        print("** settings **")
        print("number of games = ", Config.n_games)
        print("learning rate (alpha) = ", Config.lr)
        print("epsilon start = ", Config.epsilon)
        print("epsilon minimum = ", Config.eps_min)
        print("epsilon decay = ", Config.eps_decay)
        print("gamma = ", Config.gamma)
        print("batch_size = ", Config.batch_size)
        print("environment = ", Config.env_name)
        print("output graph located in ", Config.figure_file)
        print("*"*100)

