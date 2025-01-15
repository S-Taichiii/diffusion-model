import os
import datetime
import torch
import matplotlib.pyplot as plt

class Utils:
    @staticmethod
    def recordResult(output_path: str="./result", model=None, losses=None, **kwargs):
        try:
            now = datetime.datetime.now()
            dir_name: str = now.strftime(now.strftime("%Y_%m_%d_%H_%M"))
            dir_path: str = output_path + "/" + dir_name
            file_name: str = "record.txt"
            file_path = os.path.join(dir_path, file_name)

            os.makedirs(dir_path, exist_ok=True)

            # ハイパーパラメーターをテキストファイルに書き込み
            with open(file_path, 'w', encoding='utf-8') as f:
                for key, value in kwargs.items():
                    if key == "learning_time":
                        f.write(f'{key} : {value} (s)\n')
                        continue
                    f.write(f'{key} : {value}\n')

            # 学習済みモデルのパラメーターの保存
            if model: Utils.saveModelParameter(dir_path, model)

            # lossのグラフを保存
            if losses: Utils.saveLossToGraph(dir_path, losses)

        except Exception as e:
            print(f'エラーが発生しました：{e}')

    @staticmethod
    def saveModelParameter(dir_path: str, model):
        output_path: str = dir_path + "/trained_para.pth"
        torch.save(model.state_dict(), output_path) 

    @staticmethod
    def saveLossToGraph(dir_path, losses):
        file_path = dir_path + "/loss.png" 

        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(file_path)

    @staticmethod
    def show_images(images, rows=2, cols=10):
        fig = plt.figure(figsize=(cols, rows), facecolor='gray')
        i = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, i + 1)
                plt.imshow(images[i])
                plt.axis("off") # 縦軸、横軸を非表示にする
                i += 1

        plt.show()

    @staticmethod
    def show_image(images):
        fig = plt.figure(facecolor='gray')
        plt.imshow(images[0])
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
#    now = datetime.datetime.now()
#    print(now.strftime("%Y_%m_%d_%H:%M")) 
#    print(type(now.strftime("%Y_%m_%d_%H:%M")))

    batch_size = 128
    num_timesteps = 1000
    epochs = 100
    lr = 1e-3
    learning_time = 10000

    Utils.recordResult(batch_size=batch_size, num_timesteps = num_timesteps, epochs=epochs, lr = lr, learning_time=learning_time)