# DOFBOT 자산 안내

이 저장소에는 NVIDIA / Isaac Sim용 USD 바이너리를 포함하지 않습니다.

## 권장 절차

1. reference repository에서 `dofbot.urdf`를 다운로드합니다.
2. Isaac Sim 5.1을 실행합니다.
3. URDF를 instanceable USD로 변환합니다.
4. 변환한 파일을 아래 위치에 저장합니다.

   `dofbot_rl/assets/dofbot/dofbot_instanceable.usd`

## 주의사항

- 이 task는 로봇 articulation root 이름이 `Dofbot`인 것을 가정합니다.
- end-effector body 이름은 기본적으로 `link5`를 사용합니다.
- 변환한 USD의 body 이름이 다르면 `tasks/dofbot_reach_cfg.py`에서 `ee_body_name`을 수정하세요.

