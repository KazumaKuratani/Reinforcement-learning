# %%
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms





# %%
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform = transforms.ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform = transforms.ToTensor()
)
# %%
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)





# %%
#nn.Moduleは「全自動の管理パッケージ」のような存在。
#管理作業を自動化する機能が詰まっている
class NeuralNetwork(nn.Module):
    def __init__(self):
        #super().__init__()と書くことで自分(NeuralNetwork)をPytorch用の魔法がかかった特別なクラスとして正式に登録する
        #Pythonの仕組み上、自分のクラスで __init__（初期化メソッド）を書くと、親クラス（nn.Module）が元々持っていた初期化処理が上書きされて消えてしまう。
        super(NeuralNetwork,self).__init__()
        
        #画像処理において多次元のデータを1列のベクトルに平らにならす処理を行っている
        #もともとの画像データは28*28だがこれを784のベクトルに直す
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self,x):
        x = self.flatten(x)
        
        #ここのlogitsは「まだ確率になっていない、生の数値」。確率（ソフトマックス）にする前の生の値。
        #この理由はPytorchで学習を行う際、誤差を計算するnn.CrossEntropyLossは入力としてlogitsを受け取り、その内部でSoftmaxを計算するという仕組みになっている。
        #計算効率的にSoftmax と CrossEntropyはまとめて計算した方が速い(これは確かに誤差逆伝播でもそうだよね)
        #linear_relu_stackからnn.Sequentialが呼ばれ、nn.Sequentialにまとめておけばび出しで中に入っている全レイヤーを順番に実行してくれる
        logits = self.linear_relu_stack(x)
        return logits
    
    
    
    

# %%
import torch

#[真のときの値] if [条件式] else [偽の時の値]
#cuda が使えるならcuda(GPU)を使い、ないならCPUを使う
device = "cuda" if torch.cuda.is_available() else "cpu"

#NeuralNetwork(): モデルのインスタンス（実体）を作り、この瞬間、重みWやバイアスbは一時的にCPU上のメモリ（RAM） に確保される。
#to(device):ここでは俺のPCはCPUだからそのままCPUにとどまる
model = NeuralNetwork().to(device)




# %%

#損失関数を定義する、ここでは多クラス分類だから交差エントロピー誤差を利用する
loss_fn = nn.CrossEntropyLoss()




# %%
learning_rate = 1e-3
#torch.optim.最適化アルゴリズムの名称(インスタンス化したmodel.parameters()、lr=learning_rateを設定)
#super().__init__() は、重みや層を記録するための 「台帳（辞書）」 を作成している。
#model.parameters() は、その台帳を 「隅々までチェック」 して重みを集めている。
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)




# %%
def train_loop(dataloader, model ,loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    #enumerate()は常に(インデックス、中身)というペアを返す。今回のdataloaderの場合は中身自体が(画像データ、ラベル)というペアになっているから（インデックス,(画像データ、ラベル)）
    #for x,data in enumerate(dataloader):     enumerateは列挙するという意味
        #y,z = data    としてもいい
    for batch, (X,y) in enumerate(dataloader):
        
        #model(X): forward メソッドが呼ばれ、現在の重みを使って予測値（Logits）を計算する。
        pred = model(X)
        
        #loss_fn(pred, y): 予測値と正解 y を比較し、「どれくらい間違っているか」を一つの数値（loss）として出す。
        loss = loss_fn(pred,y)
        
        #PyTorchは放っておくと勾配を足し算し続けてしまうため、毎回リセットが必要
        optimizer.zero_grad()
        
        #backward()で誤差lossを元に、backward()処理（勾配計算）をPyTorchが逆伝播で自動で行ってくれる
        loss.backward()
        
        #stepで計算された勾配を使って実際にモデルの重みW,bを更新する
        optimizer.step()
        
        if batch % 100 == 0:
            loss,current = loss.item(), batch*len(X)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")







# %%


#このtest_loop()はどれくらい正解できてるかを測定するだけだから重みを更新しない

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    
    #PyTorchは通常計算を行うたびに「計算した情報」を裏側でメモしているからここではそのメモを完全にオフしている
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            
            #.item() を使うことで、純粋な Pythonの数値（float） に変換し、身軽な状態で合計している。変換しない場合は「PyTorchのテンソル形式」で受け取っていて、不要な情報も引き連れている。
            test_loss += loss_fn(pred, y).item()
            
            #predは各数字に対するスコアのリスト
            #(pred.argmax(1) == y)で[True, False, True, ...] という真偽値のリスト（テンソル）になる
            #.type(torch.float)でPyTorchではTrue/Falseのまま計算はできないため1.0と0.0の数値に変換している
            #.item() で、PyTorchの世界からPythonの数値として取り出しcorrectに加算する
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")






# %%
epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# %%
torch.save(model.state_dict(), "model_weights_2layers.pth")
print("Saved PyTorch Model State to model_weights_2layer.pth")