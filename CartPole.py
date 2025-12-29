

# %%
import gymnasium as gym
env = gym.make("CartPole-v0")
# %%
state = env.reset()
print(state)

action_space = env.action_space
print(action_space)
# %%
import numpy as np
import random
from collections import deque
# %%
class ReplayBuffer:
    def __init__(self,buffer_size,batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        data = (state,action,reward,next_state,done)
        self.buffer.append(data)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer,self.batch_size)
        
        
        #stack→同じ形の配列を次元を積み重ねて新しい配列（テンソル）を作る。
        #array→actionとrewardではスカラー値を受け取っていてそれを1次元配列に変換している。（スカラー値→1次元配列ならarrayで十分だよねという話)
        #state→これがニューラルネットワークの入力になるよね、ニューラルネットワークの入力は"状態"のみであって、("状態"、"行動)ではないことに注意
        state      = np.stack([x[0] for x in data])
        action     = np.array([x[1] for x in data])
        reward     = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        # doneは学習時に計算しやすいよう整数(0 or 1)にしておくと便利
        done       = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done
        
# %%
from torch import nn    
import copy  
# %%
#class Name()とすることで"継承"（"引数"とは違うことに注意）する。最初からnn.Moduleが持っている機能を使えるようになる。
#継承することで「NeuralNetwork という名前で新しく定義するけど、そのベース（型紙）には nn.Module を使ってくれ！」という指示出しになる。
class QNetwork(nn.Module):
    def __init__(self,state_size,action_size):
        #nn.Module:基底クラス（スーパークラス、他のクラスが継承するための親クラス）
        #nn.Moduleを使うことで➀パラメータ管理➁GPU対応➂保存機能　など色々提供してくれる。
        #super().__init__()と書くことで自分(NeuralNetwork)をPytorch用の魔法がかかった特別なクラスとして正式に登録する
        #Pythonの仕組み上、自分のクラスで __init__（初期化メソッド）を書くと、親クラス（nn.Module）が元々持っていた初期化処理が上書きされて消えてしまう。
        super(QNetwork,self).__init__()
        
        
        #NNの入力：状態　出力：行動価値関数(行動の分だけ)
        #ここではNNの入力(つまり状態)は4つでカートの位置、カートの速度、棒の角度、棒の角速度　になる
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size,64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )
        
    def forward(self,x):
        y = self.linear_relu_stack(x)
        return y
        
# %%
import torch
import torch.optim as optim
# %%
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.action_size = action_size
        self.epsilon = 1.0           # 最初は100%ランダムに動く
        self.epsilon_decay = 0.995   # 1エピソードごとに減衰させる割合
        self.epsilon_min = 0.01      # これ以上は小さくしない限界値
        
        # ① メインネットワーク（常に学習・更新される）
        self.qnet = QNetwork(state_size, action_size)
        
        # メインネットワークを丸ごとコピーして作成する
        #deepcopyすることで完全に別のコピーを作る(copyだけだと上書きされてしまう)
        self.qnet_target = copy.deepcopy(self.qnet)
        
        # 最適化手法（Adamなど）はメインネットワークにだけ適用する
        #self.qnet.parameters()でnn.Module の魔法で、メインネットワーク内の全レイヤーの重みを自動でかき集めている
        #一旦Adam適用する
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        
        
    #メインネットワークの成果をターゲットネットワークに反映させる
    def sync_qnet(self):
        #state_dict()でネットワークが持っている全ての重みとバイアスを辞書形式で返してくれる
        #load_state_dict()で受け取った辞書の中身を自分自身の各レイヤーに上書きする
        #deepcopy VS load_state_dict →load_state_dict:すでにあるオブジェクトの中身の数値だけを置き換える（速い)
        #deepcopy:全く新しいオブジェクトをメモリ上に作り直す(重い処理)
        #例えば1,000ステップに一回だけsync_qnetを呼び出し、メインの成果をターゲットに反映させる
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        
    def get_action(self,state):
        #np.random.rand(): 0から1の間のランダムな数字を1つ出す、rand=乱数
        if np.random.rand() < self.epsilon:
            
            #np.random.choice は、本来はリストなどから要素をランダムに選ぶ関数だが、引数に整数を渡すと、自動的に「0からその数字未満の整数」の範囲から選んでくれる。
            return np.random.choice(self.action_size)
        else:
            #ニューラルネットワークは
            #np.newaxis,:　でデータの値はそのままで次元ふ1つ増やすというお作法
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
            
            #Qnetworkのforwardが動き出す
            #ここめちゃくちゅ重要
            #qnet.forward(state)と書いてはダメ
            #pythonではクラスのインスタンスを関数のように呼び出したときに実行される特別な__call__という特別なメソッドがある
            #nn.Module(基底クラス、親クラス)の内部では
            # nn.Module の内部（簡略化）ではdef __call__(self,x):➀前処理➁forward➂後処理という1連のパッケージが自動的に実行される
            #self.qnet.forward(state)と書いてしまうとpytorchが裏側で行っている機能をスキップする
            qs = self.qnet(state)
            # .item() をつけることで、tensor(0) が 0 に変換される
            #Gym は「Tensor」が嫌いだから
            return qs.argmax().item()
        
        
        
    def update(self, state, action, reward, next_state, done):
    #PyTorchでの重みはデフォルトで32ビット浮動小数点数になっている
    #32ビットの方が(64ビットに比べ)高速処理が可能、あとはAI学習には最適
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        #squeeze=絞る、押しつぶす　
        #tensor.unsqueeze(d: int)は「テンソルのd階目に要素数1の次元を挿入する」 
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(1) #unsqueese(1)は1次元の配列を2次元の縦ベクトルに変換する
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)


        # 2. 現在のQ値（予測値）を計算
        # ネットワークが出した[左のQ値, 右のQ値]から、実際に取った行動(action)の方だけを抽出する
        qs = self.qnet(state)
        
        #gather()は指定した次元に沿って、インデックスに一致する値を拾う関数
        #qsはactionごとに行動価値関数が入っている(ここではバッチサイズ×行動数だから...)
        q_value = qs.gather(1, action)

    # 3. ターゲット（正解に近い値）を計算
    # ターゲット計算では勾配（学習）は不要
    #with構文によってある特定の範囲内だけ特別な設定やルールを適用する
        with torch.no_grad(): 
            
        # 次の状態での最大Q値を"ターゲットネットワーク"に予測させる。つまり次の状態の価値関数は"少し古い重み"を使って正解ラベルに利用する
            next_qs = self.qnet_target(next_state)
            
            #PyTorchのmax関数は(最大値そのもの,最大値がどこにあったかのインデックス)の2つを返す
            next_q_value = next_qs.max(1)[0].unsqueeze(1)
        
        # ベルマン方程式のターゲット： 報酬 + (時間割引率 * 次の最大Q値) 
        #最後は報酬のみ、だから最後はdone=1でrewardだけ受け取る
            target = reward + (1 - done) * self.gamma * next_q_value

    # 4. 誤差を計算（MSELoss: 平均二乗誤差）
        loss = nn.functional.mse_loss(q_value, target)

    # 5. ネットワークの更新（ここが「学習」の3点セット！）
        #PyTorchは計算した重みの勾配を消さずにどんどん足し算していく可能性があるため、一回勾配をリセットする
        self.optimizer.zero_grad() 
        loss.backward()            
        self.optimizer.step()      
# %%
batch_size = 32
episodes = 500
target_update_interval = 10
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
buffer = ReplayBuffer(buffer_size=10000,batch_size=32)




# --- メインループ ---
reward_history = []
for episode in range(episodes):
    state, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # 1. 行動を選択
        action = agent.get_action(state)
        
        # 2. 環境を動かす
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 3. 経験をメモリに保存
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        # 4. 学習（バッファに十分データが溜まったら）
        if len(buffer) > batch_size:
            s_batch, a_batch, r_batch, ns_batch, d_batch = buffer.get_batch()
            agent.update(s_batch, a_batch, r_batch, ns_batch, d_batch)
            
    reward_history.append(total_reward)       
    # 5. ターゲットネットワークの同期
    if episode % target_update_interval == 0:
        agent.sync_qnet()
        
    print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    # イプシロンを少しずつ減らす（徐々に探索を減らして活用を増やす）
    if agent.epsilon > 0.01:
        agent.epsilon *= 0.99
# %%
# メインネットワークの重みを保存
torch.save(agent.qnet.state_dict(), "cartpole_dqn.pth")
print("モデルを保存しました。")
# %%
# 同じ構造のネットワークを作り、保存した重みを流し込む
agent.qnet.load_state_dict(torch.load("cartpole_dqn.pth"))
agent.qnet.eval() # 評価モード（学習を止める）に設定
# %%# 再生用に render_mode を 'human' に設定して環境を再作成
env_render = gym.make('CartPole-v1', render_mode='human')
state, _ = env_render.reset()
done = False

while not done:
    # 学習後のAIに行動を選ばせる（イプシロンは0にする）
    state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
    with torch.no_grad():
        qs = agent.qnet(state_tensor)
        action = qs.argmax().item()
    
    state, reward, terminated, truncated, _ = env_render.step(action)
    done = terminated or truncated
    env_render.render() # 画面に描画

env_render.close()

# %%
import matplotlib.pyplot as plt


plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Learning Curve')
plt.show()
# %%
