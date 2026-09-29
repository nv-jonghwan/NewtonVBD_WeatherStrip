# 외부 프로젝트 및 자산 고지

본 저장소의 코드와 문서는 Apache License 2.0으로 제공합니다. 외부 엔진과 로봇 자산의 라이선스는 각각의 원저작권자 조건을 따릅니다. 아래 로봇 USD 원본은 이 Git 저장소에 재배포하지 않으며, `scripts/fetch_assets.py`가 `assets/manifest.json`에 기록된 공식 출처에서 받아 SHA-256을 검증합니다.

| 구성 요소 | 출처 | 적용 조건 |
| --- | --- | --- |
| Newton | [newton-physics/newton](https://github.com/newton-physics/newton) | Apache-2.0 |
| MuJoCo / MuJoCo Warp | [MuJoCo](https://github.com/google-deepmind/mujoco), [MJWarp](https://github.com/google-deepmind/mujoco_warp) | Apache-2.0 |
| NVIDIA Warp | [NVIDIA/warp](https://github.com/NVIDIA/warp) | Apache-2.0 |
| Isaac Lab | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | BSD-3-Clause 및 각 구성 요소 고지 |
| Isaac Sim | [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) | NVIDIA가 제공하는 배포물의 사용 조건 |
| FANUC CRX-10iA/L USD | NVIDIA Isaac Sim 6.0 robot assets | [NVIDIA 3D Content Sharing Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/3d-content-sharing-agreement/) 및 자산별 고지 |
| Robotiq 2F-85 USD | [NVIDIA SimReady Foundation](https://github.com/NVIDIA/simready-foundation/tree/0ed0dfbc539c9de99289771bd6848effe3ef5779/sample_content/common_assets/robots_general/Robotiq/2F-85/simready_isaac_usd) | 저장소 [LICENSE.txt](https://github.com/NVIDIA/simready-foundation/blob/0ed0dfbc539c9de99289771bd6848effe3ef5779/LICENSE.txt) 및 자산별 고지 |

`docs/media/`의 그림은 이 예제의 실제 Isaac Lab 실행 화면입니다. 그림 속 로봇 디자인과 상표의 권리는 각 권리자에게 있습니다. FANUC, Robotiq, NVIDIA 또는 Google DeepMind의 공식 제품·인증·보증을 의미하지 않습니다.

Newton의 공개 솔버 API와 결합 예제를 참고해 시뮬레이션을 구성했습니다. 이 저장소는 엔진 소스 자체를 포함하거나 수정하지 않습니다. 새로 생성한 웨더스트립 USD는 공식 SimReady 인증 자산이 아니며, 예시 물성을 사용하는 검증 가능한 참조 구현입니다.
