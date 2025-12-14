#!/usr/bin/env python3
"""
PPO 알고리즘을 사용한 자율주행 레이싱 카 학습 스크립트
학습 후 ONNX 형식으로 모델을 변환하여 라즈베리파이 배포 준비
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import onnx
import onnxruntime as ort
from env import RandomTrackEnv
import matplotlib.pyplot as plt
from collections import deque


def create_env(render_mode=None):
    """
    환경 생성 함수 (벡터화를 위해 필요)

    Args:
        render_mode: 렌더링 모드 ("human", "rgb_array", None)
    """
    env = RandomTrackEnv(
        num_ultrasonic_sensors=2,
        sensor_angles=[-15, 75],
        sensor_max_range=150.0,
        sensor_noise_std_dev=2.0,  # Sim-to-Real: 센서 노이즈
        track_width_min=40.0,
        track_width_max=47.0,
        track_length_min=1000.0,
        track_length_max=4000.0,
        friction_variation=0.1,  # Sim-to-Real: 마찰 변동
        render_mode=render_mode,
    )
    return env


class LearningCurveCallback(BaseCallback):
    """
    학습 곡선을 실시간으로 그리는 콜백
    """
    def __init__(self, window_size=100, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.timesteps = []
        self.mean_rewards = []
        self.fig = None
        self.ax1 = None
        self.ax2 = None

    def _on_training_start(self) -> None:
        # 그래프 초기화
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Learning Progress', fontsize=14, fontweight='bold')

        # 리워드 그래프
        self.ax1.set_xlabel('Episode', fontsize=10)
        self.ax1.set_ylabel('Reward', fontsize=10)
        self.ax1.set_title('Episode Rewards (Moving Average)', fontsize=12)
        self.ax1.grid(True, alpha=0.3)

        # 에피소드 길이 그래프
        self.ax2.set_xlabel('Episode', fontsize=10)
        self.ax2.set_ylabel('Steps', fontsize=10)
        self.ax2.set_title('Episode Length (Moving Average)', fontsize=12)
        self.ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def _on_step(self) -> bool:
        # Monitor에서 에피소드 정보 가져오기
        # stable-baselines3의 Monitor는 에피소드가 끝날 때 'episode' 키를 infos에 추가
        infos = self.locals.get('infos', [])
        for info in infos:
            if isinstance(info, dict) and 'episode' in info:
                episode_info = info['episode']
                episode_reward = episode_info.get('r', 0)
                episode_length = episode_info.get('l', 0)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                # 이동 평균 계산
                if len(self.episode_rewards) > 0:
                    mean_reward = np.mean(self.episode_rewards)
                    mean_length = np.mean(self.episode_lengths)

                    self.timesteps.append(self.num_timesteps)
                    self.mean_rewards.append(mean_reward)

                    # 그래프 업데이트 (에피소드가 끝날 때마다)
                    self._update_plots()

        return True

    def _update_plots(self):
        """그래프 업데이트"""
        if len(self.episode_rewards) == 0:
            return

        # 리워드 그래프
        self.ax1.clear()
        episodes = range(len(self.episode_rewards))
        self.ax1.plot(episodes, list(self.episode_rewards),
                     'b-', alpha=0.3, linewidth=1, label='Episode Reward')
        if len(self.mean_rewards) > 0:
            self.ax1.plot(range(len(self.mean_rewards)), self.mean_rewards,
                         'r-', linewidth=2, label=f'Moving Avg (n={len(self.episode_rewards)})')
        self.ax1.set_xlabel('Episode', fontsize=10)
        self.ax1.set_ylabel('Reward', fontsize=10)
        self.ax1.set_title(f'Episode Rewards (Latest: {self.episode_rewards[-1]:.2f}, '
                          f'Avg: {np.mean(self.episode_rewards):.2f})', fontsize=12)
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # 에피소드 길이 그래프
        self.ax2.clear()
        self.ax2.plot(episodes, list(self.episode_lengths),
                     'g-', alpha=0.3, linewidth=1, label='Episode Length')
        if len(self.episode_lengths) > 0:
            mean_lengths = [np.mean(list(self.episode_lengths)[:i+1])
                           for i in range(len(self.episode_lengths))]
            self.ax2.plot(episodes, mean_lengths,
                         'orange', linewidth=2, label=f'Moving Avg')
        self.ax2.set_xlabel('Episode', fontsize=10)
        self.ax2.set_ylabel('Steps', fontsize=10)
        self.ax2.set_title(f'Episode Length (Latest: {self.episode_lengths[-1]}, '
                           f'Avg: {np.mean(self.episode_lengths):.1f})', fontsize=12)
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def _on_training_end(self) -> None:
        # 최종 그래프 저장
        if self.fig is not None:
            os.makedirs("logs", exist_ok=True)
            self.fig.savefig("logs/learning_curve.png", dpi=150, bbox_inches='tight')
            print("\n[학습 곡선] 그래프가 저장되었습니다: logs/learning_curve.png")
            # 그래프는 열어둠 (사용자가 확인할 수 있도록)


class RenderCallback(BaseCallback):
    """
    주기적으로 환경을 렌더링하는 콜백
    """
    def __init__(self, render_freq=1000, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.render_env = None

    def _on_training_start(self) -> None:
        # 학습 시작 시 렌더링용 환경 생성
        self.render_env = create_env(render_mode="human")

    def _on_step(self) -> bool:
        # 주기적으로 렌더링
        if self.n_calls % self.render_freq == 0 and self.render_env is not None:
            try:
                # 현재 정책으로 한 에피소드 실행 및 렌더링
                obs, _ = self.render_env.reset()
                done = False
                steps = 0
                max_steps = 200  # 최대 200 스텝

                print(f"\n[시각화] 학습 진행 상황 렌더링 (타임스텝: {self.num_timesteps})")

                while not done and steps < max_steps:
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, info = self.render_env.step(action)
                    done = terminated or truncated
                    steps += 1
                    self.render_env.render()

                if done:
                    print(f"[시각화] 에피소드 완료 (스텝: {steps}, 충돌: {info.get('collision', False)})")
                else:
                    print(f"[시각화] 최대 스텝 도달 (스텝: {steps})")

            except Exception as e:
                print(f"[시각화] 렌더링 오류: {e}")

        return True

    def _on_training_end(self) -> None:
        # 학습 종료 시 환경 정리
        if self.render_env is not None:
            self.render_env.close()
            self.render_env = None


class ActorWrapper(torch.nn.Module):
    """
    Stable-Baselines3의 Actor 네트워크를 ONNX 변환을 위한 래퍼
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        """
        관측값으로부터 액션 예측
        """
        # policy의 _predict 메서드 사용
        # 또는 직접 actor 네트워크 호출
        if hasattr(self.policy, 'mlp_extractor'):
            # Feature extraction
            features = self.policy.extract_features(observation)
            # Action prediction
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
            action = self.policy.action_net(latent_pi)
            return action
        else:
            # Fallback: policy의 forward 사용
            return self.policy(observation)


def export_to_onnx(model, save_path="racing_car_policy.onnx", input_size=4):
    """
    Stable-Baselines3 PPO 모델을 ONNX 형식으로 변환

    Args:
        model: 학습된 PPO 모델
        save_path: ONNX 모델 저장 경로
        input_size: 입력 크기 (초음파 센서 수 + 1)
    """
    print(f"\n[ONNX 변환] 모델을 ONNX 형식으로 변환 중...")

    # PyTorch 모델 추출
    policy = model.policy

    # 모델을 평가 모드로 설정
    policy.eval()

    # Actor 래퍼 생성
    actor_wrapper = ActorWrapper(policy)
    actor_wrapper.eval()

    # 더미 입력 생성 (정규화된 관측값)
    dummy_input = torch.zeros((1, input_size), dtype=torch.float32)

    # ONNX로 변환
    try:
        # 먼저 일반적인 방법 시도
        try:
            torch.onnx.export(
                actor_wrapper,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=11,  # ONNX opset 버전
                do_constant_folding=True,
                input_names=['observation'],
                output_names=['action'],
                dynamic_axes={
                    'observation': {0: 'batch_size'},
                    'action': {0: 'batch_size'}
                },
                verbose=False
            )
        except Exception as e1:
            print(f"[ONNX 변환] 첫 번째 방법 실패, 대체 방법 시도: {e1}")
            # 대체 방법: action_net만 직접 변환
            if hasattr(policy, 'action_net'):
                # Feature extractor와 action_net을 결합한 간단한 네트워크
                class SimpleActor(torch.nn.Module):
                    def __init__(self, policy):
                        super().__init__()
                        self.features_extractor = policy.features_extractor
                        if hasattr(policy, 'mlp_extractor'):
                            self.mlp_extractor = policy.mlp_extractor
                        self.action_net = policy.action_net

                    def forward(self, obs):
                        features = self.features_extractor(obs)
                        if hasattr(self, 'mlp_extractor'):
                            latent_pi = self.mlp_extractor.forward_actor(features)
                        else:
                            latent_pi = features
                        return self.action_net(latent_pi)

                simple_actor = SimpleActor(policy)
                simple_actor.eval()

                torch.onnx.export(
                    simple_actor,
                    dummy_input,
                    save_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['observation'],
                    output_names=['action'],
                    dynamic_axes={
                        'observation': {0: 'batch_size'},
                        'action': {0: 'batch_size'}
                    },
                    verbose=False
                )
            else:
                raise e1

        print(f"[ONNX 변환] 성공: {save_path}")

        # ONNX 모델 검증
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("[ONNX 변환] 모델 검증 완료")

        # ONNX Runtime으로 테스트
        ort_session = ort.InferenceSession(save_path)
        test_input = np.random.randn(1, input_size).astype(np.float32)
        outputs = ort_session.run(None, {'observation': test_input})
        print(f"[ONNX 변환] 추론 테스트 성공: 출력 shape = {outputs[0].shape}")

        return True

    except Exception as e:
        print(f"[ONNX 변환] 오류 발생: {e}")
        print("[ONNX 변환] 팁: PyTorch 모델을 직접 사용하거나 수동으로 변환을 시도하세요.")
        import traceback
        traceback.print_exc()
        return False


def train_ppo(
    total_timesteps=1_000_000,
    learning_rate=5e-4,  # 학습률 증가 (더 빠른 학습)
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,  # 엔트로피 계수 증가 (더 많은 탐험)
    vf_coef=0.5,
    max_grad_norm=0.5,
    save_freq=10000,
    eval_freq=50000,
    n_eval_episodes=10,
    render_eval=True,  # 평가 중 렌더링 여부
    render_freq=5000,  # 주기적 렌더링 빈도 (None이면 비활성화)
):
    """
    PPO 알고리즘으로 모델 학습

    Args:
        total_timesteps: 총 학습 타임스텝
        learning_rate: 학습률
        n_steps: 각 업데이트마다 수집할 스텝 수
        batch_size: 배치 크기
        n_epochs: 각 업데이트마다 수행할 에포크 수
        gamma: 할인 계수
        gae_lambda: GAE 람다
        clip_range: PPO 클리핑 범위
        ent_coef: 엔트로피 계수
        vf_coef: 가치 함수 손실 계수
        max_grad_norm: 그래디언트 클리핑
        save_freq: 체크포인트 저장 빈도
        eval_freq: 평가 빈도
        n_eval_episodes: 평가 에피소드 수
    """
    print("=" * 60)
    print("PPO 학습 시작")
    print("=" * 60)

    # 디렉토리 생성
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 환경 생성
    print("\n[환경 설정] 랜덤 트랙 환경 생성 중...")
    env = create_env()

    # 모니터링 래퍼 (로그 기록용)
    env = Monitor(env, "logs/")

    # 벡터화된 환경 (PPO는 벡터화된 환경을 사용)
    env = DummyVecEnv([lambda: env])

    # 평가용 환경 생성 (렌더링 옵션)
    eval_env = create_env(render_mode="human" if render_eval else None)
    eval_env = Monitor(eval_env, "logs/eval/")
    eval_env = DummyVecEnv([lambda: eval_env])

    # PPO 모델 생성
    print("\n[모델 생성] PPO 모델 초기화 중...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log="logs/tensorboard/",
        device="auto",  # GPU가 있으면 자동으로 사용
    )

    print(f"[모델 생성] 완료")
    print(f"  - 정책 네트워크: MLP")
    print(f"  - 관측 공간: {env.observation_space}")
    print(f"  - 액션 공간: {env.action_space}")

    # 콜백 설정
    print("\n[콜백 설정] 학습 콜백 설정 중...")

    # 체크포인트 콜백 (정기적으로 모델 저장)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="models/checkpoints/",
        name_prefix="ppo_racing_car",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # 평가 콜백 (최고 성능 모델 저장)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best/",
        log_path="logs/eval/",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=render_eval,  # 평가 중 렌더링
    )

    # 콜백 리스트
    callbacks = [checkpoint_callback, eval_callback]

    # 학습 곡선 콜백 추가
    learning_curve_callback = LearningCurveCallback(window_size=100)
    callbacks.append(learning_curve_callback)
    print("[시각화] 학습 곡선 그래프 활성화")

    # 주기적 렌더링 콜백 추가 (옵션)
    if render_freq is not None and render_freq > 0:
        render_callback = RenderCallback(render_freq=render_freq)
        callbacks.append(render_callback)
        print(f"[시각화] 주기적 렌더링 활성화 (매 {render_freq} 스텝마다)")

    # 학습 시작
    print("\n" + "=" * 60)
    print("학습 시작")
    print(f"  - 총 타임스텝: {total_timesteps:,}")
    print(f"  - 학습률: {learning_rate}")
    print(f"  - 배치 크기: {batch_size}")
    print(f"  - 에포크 수: {n_epochs}")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        print("\n" + "=" * 60)
        print("학습 완료!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[경고] 사용자에 의해 학습이 중단되었습니다.")
        print("현재까지의 모델을 저장합니다...")

    # 최종 모델 저장
    final_model_path = "models/ppo_racing_car_final"
    print(f"\n[모델 저장] 최종 모델 저장 중: {final_model_path}")
    model.save(final_model_path)
    print("[모델 저장] 완료")

    # ONNX 변환
    print("\n" + "=" * 60)
    print("ONNX 변환 시작")
    print("=" * 60)

    # 입력 크기 확인 (초음파 센서 수 + 1)
    input_size = env.observation_space.shape[0]
    onnx_path = "models/racing_car_policy.onnx"

    success = export_to_onnx(model, onnx_path, input_size)

    if success:
        print(f"\n[완료] ONNX 모델이 저장되었습니다: {onnx_path}")
        print("이 모델을 라즈베리파이에서 사용할 수 있습니다.")
    else:
        print("\n[경고] ONNX 변환에 실패했습니다.")
        print("PyTorch 모델을 직접 사용하거나 변환을 다시 시도하세요.")

    # 환경 정리
    env.close()
    eval_env.close()

    return model


def test_model(model_path="models/best/best_model", n_episodes=5, render=True):
    """
    학습된 모델 테스트

    Args:
        model_path: 모델 경로
        n_episodes: 테스트 에피소드 수
        render: 렌더링 여부
    """
    print("\n" + "=" * 60)
    print("모델 테스트")
    print("=" * 60)

    # 모델 로드
    print(f"\n[모델 로드] {model_path} 로드 중...")
    model = PPO.load(model_path)

    # 환경 생성 (렌더링 활성화) - 학습 환경과 동일한 설정 사용
    env = create_env(render_mode="human" if render else None)

    # 테스트 실행
    total_rewards = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\n[에피소드 {episode + 1}/{n_episodes}] 시작")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            if steps > 1000:  # 무한 루프 방지
                break

        total_rewards.append(total_reward)
        print(f"[에피소드 {episode + 1}] 완료 - 총 리워드: {total_reward:.2f}, 스텝: {steps}")

    env.close()

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"평균 리워드: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"최고 리워드: {np.max(total_rewards):.2f}")
    print(f"최저 리워드: {np.min(total_rewards):.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO 학습 스크립트")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="실행 모드: train (학습) 또는 test (테스트)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="총 학습 타임스텝 (기본값: 1,000,000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best/best_model",
        help="테스트할 모델 경로",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="테스트 시 렌더링 비활성화",
    )
    parser.add_argument(
        "--render-eval",
        action="store_true",
        default=True,
        help="평가 중 렌더링 활성화 (기본값: True)",
    )
    parser.add_argument(
        "--no-render-eval",
        action="store_false",
        dest="render_eval",
        help="평가 중 렌더링 비활성화",
    )
    parser.add_argument(
        "--render-freq",
        type=int,
        default=5000,
        help="주기적 렌더링 빈도 (스텝 수, 0이면 비활성화, 기본값: 5000)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        # 학습 실행
        model = train_ppo(
            total_timesteps=args.timesteps,
            render_eval=args.render_eval,
            render_freq=args.render_freq if args.render_freq > 0 else None,
        )

        # 학습 후 간단한 테스트
        print("\n" + "=" * 60)
        print("학습 완료 후 테스트")
        print("=" * 60)
        test_model("models/ppo_racing_car_final", n_episodes=3, render=not args.no_render)

    elif args.mode == "test":
        # 테스트만 실행
        test_model(args.model_path, n_episodes=5, render=not args.no_render)

