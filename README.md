index.html              프로토타입 앱 (브라우저만으로 동작, GitHub Pages로 게시)
                        - 관리자 화면: 대상 선택, 문구 작성, 푸시 발송, 캠페인 현황, A/B 결과
                        - 고객 화면: 홈, 알림함, 혜택 안내, 설정(수신 동의, 큰 글씨)
data/generate_data.py   1주차: 합성 앱 로그 생성 (고객 2만 명, 이벤트 약 11만 건, 수신 동의 열 포함)
model/train.py          2주차: 가입 7일째 이탈 예측 모델 학습·평가 -> 위험 점수 저장
tools/export_demo.py    DB의 테스트 고객과 위험 점수를 index.html 안의 데이터로 내보내기
app/app.py              서버(Flask) 버전: index.html 과 같은 기능. 발송·열람 기록과 수신 동의를 DB에 저장하고
                        여러 기기가 공유한다 (PC에서 발송 -> 다른 기기의 고객 화면에 알림 도착)
requirements.txt        필요한 파이썬 패키지
