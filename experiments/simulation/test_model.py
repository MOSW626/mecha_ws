#!/usr/bin/env python3
"""
학습된 모델을 테스트하는 간단한 스크립트
"""

import argparse
from train import test_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델 테스트")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_racing_car_final",
        help="테스트할 모델 경로 (확장자 없이)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="테스트할 에피소드 수",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="렌더링 비활성화 (더 빠름)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("모델 테스트 시작")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"에피소드 수: {args.episodes}")
    print(f"렌더링: {'비활성화' if args.no_render else '활성화'}")
    print("=" * 60)
    print()

    test_model(args.model, n_episodes=args.episodes, render=not args.no_render)

