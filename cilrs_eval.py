import os
import torch
import yaml

from carla_env.env import Env

from torchvision.transforms import ToTensor, Resize, Compose, Normalize


class Evaluator():
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent = self.load_agent()

    def load_agent(self, ):
        # Your code here
        # model_logs/cilrs_model_lr_2e-4_branched_dropout_0.5_img224_epoch_6.ckpt
        #  this one works well but somtimes stops
        # Best MSE Branched: 'model_logs/cilrs_model_cilr_hyperopt_440050371_epoch_20.ckpt'
        path = 'model_logs/test/cilrs_model.ckpt'# 'model_logs/cilrs_model_cilr_hyperopt_627198925_l1_epoch_26.ckpt' # 'model_logs/cilrs_model_cilr_hyperopt_627198925_l1_epoch_10.ckpt'
        model = torch.load(path)
        print(dir(model))
        model.eval()

        return model

    def generate_action(self, rgb, command, speed):
        # Your code here
        self.transforms = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            Resize((224, 224))
        ])

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        image = self.transforms(rgb).unsqueeze(0).to(device)
        v = torch.tensor(speed).type(torch.float).unsqueeze(0).to(device)
        c = torch.tensor(command).type(torch.long).unsqueeze(0).to(device)
        
        
        with torch.no_grad():
            throttle, steer, brake, _ = self.agent(image, v, c)
            throttle, steer, brake = float(throttle.squeeze().cpu().numpy()), float(steer.squeeze().cpu().numpy()), float(brake.squeeze().cpu().numpy())

        if brake < 0.01:
            brake = 0.0

        print(throttle, steer, brake, command)
    
        return throttle, steer, brake

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, steer, brake = self.generate_action(rgb, command, speed)
        action = {
            "throttle": throttle,
            "brake": brake,
            "steer": steer
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            state, _, is_terminal = self.env.reset()
            for i in range(5000):
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
