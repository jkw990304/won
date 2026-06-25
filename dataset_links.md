# 일반 이미지 데이터셋 조사 결과

목표: 일반 이미지로 아래 3개 클래스를 분류한다.

- `clap`: 박수 / 박수치는 사람
- `doorbell`: 초인종
- `fire_alarm`: 화재경보기 / 화재경보 시스템

## 결론

가장 바로 쓸 수 있는 공개 데이터셋은 **Open Images V7**이다.

Open Images 실제 클래스 목록에서 아래 라벨을 확인했다.

| 우리 클래스 | Open Images 라벨 | Label ID | 사용성 |
| --- | --- | --- | --- |
| `clap` | `Applause` | `/m/028ght` | 이미지 분류용으로 사용 가능 |
| `doorbell` | `Doorbell` | `/m/03wwcy` | 이미지 분류용으로 사용 가능 |
| `fire_alarm` | `Fire alarm system` | `/m/0c3f7m` | 이미지 분류용으로 사용 가능 |

중요:

- 이 세 라벨은 **이미지 분류 라벨**로 확인됨.
- YOLO처럼 위치까지 찾는 **bounding box 객체 탐지용 라벨**에는 `Doorbell`, `Fire alarm system`, `Applause`가 없음.
- bounding box 라벨에는 `Human hand` 정도만 있음.
- 따라서 지금 목표가 "이미지 1장을 보고 세 클래스 중 무엇인지 분류"라면 Open Images가 맞다.
- 목표가 "이미지 안에서 초인종/화재경보기 위치를 박스로 찾기"라면 Roboflow나 직접 라벨링이 필요하다.

## 1순위: Open Images V7

- 공식 페이지: https://storage.googleapis.com/openimages/web/index.html
- 다운로드 설명: https://storage.googleapis.com/openimages/web/download_v7.html
- 클래스 목록 CSV: https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv

Open Images 규모:

- 9M+ 이미지
- 20,638개 image-level labels
- 600개 bounding box classes

이번 프로젝트에 맞는 이유:

- `Applause`, `Doorbell`, `Fire alarm system` 라벨이 실제로 존재함.
- 세 클래스를 이미지 분류 데이터로 구성할 수 있음.
- 출처가 명확해서 공모전 보고서에 쓰기 좋음.

추천 사용 방식:

```text
Open Images에서 image-level label 기준으로
Applause / Doorbell / Fire alarm system 이미지만 추출
→ clap / doorbell / fire_alarm 폴더로 정리
→ 이미지 분류 모델 학습
```

원하는 폴더 구조:

```text
image_dataset/
  clap/
  doorbell/
  fire_alarm/
```

## 2순위: Roboflow Universe

Open Images는 이미지 분류에는 좋지만, 객체 위치 박스가 필요하면 Roboflow 쪽을 같이 봐야 한다.

- Fire alarm 검색: https://universe.roboflow.com/search?q=fire%20alarm
- Doorbell 검색: https://universe.roboflow.com/search?q=doorbell
- Clapping hands 검색: https://universe.roboflow.com/search?q=clapping%20hands
- Applause 검색: https://universe.roboflow.com/search?q=applause

추천 사용:

- `doorbell`, `fire_alarm` 객체 탐지 데이터가 필요할 때
- YOLO 형식으로 바로 내려받고 싶을 때

주의:

- Roboflow 데이터셋은 만든 사람이 제각각이라 품질 차이가 큼.
- 라이선스를 데이터셋마다 확인해야 함.
- `clap`은 객체가 아니라 행동/상황이라 Roboflow보다 Open Images의 `Applause`가 더 자연스럽다.

## 3순위: Wikimedia Commons

Open Images만으로 이미지 수가 부족하면 Wikimedia Commons 이미지를 보충 데이터로 쓸 수 있다.
라이선스가 파일별로 표시되어 있고, 실제 사진이 많아서 보고서/데모용 보충 이미지로 좋다.

### Doorbell

- Doorbells category: https://commons.wikimedia.org/wiki/Category:Doorbells
- Doorbell buttons: https://commons.wikimedia.org/wiki/Category:Doorbell_buttons
- Electric doorbells: https://commons.wikimedia.org/wiki/Category:Electric_doorbells
- Wireless doorbells: https://commons.wikimedia.org/wiki/Category:Wireless_doorbells

확인한 내용:

- `Doorbells` 카테고리에는 초인종 사진이 있고, 하위 카테고리도 있음.
- `Doorbell buttons`는 초인종 버튼 이미지가 많아서 `doorbell` 클래스 보충에 가장 적합하다.
- Commons 페이지 기준 `Doorbell buttons`는 90개 파일이 있는 하위 카테고리로 표시됨.

### Fire alarm

- Fire alarms category: https://commons.wikimedia.org/wiki/Category:Fire_alarms
- Fire alarm pull stations: https://commons.wikimedia.org/wiki/Category:Fire_alarm_pull_stations
- Manual call points: https://commons.wikimedia.org/wiki/Category:Manual_call_points
- Fire alarm notification appliances: https://commons.wikimedia.org/wiki/Category:Fire_alarm_notification_appliances
- Fire detector: https://commons.wikimedia.org/wiki/Category:Fire_detector

확인한 내용:

- `Fire alarms` 카테고리에는 200개 이상 파일이 있음.
- 하위 카테고리에 `Fire alarm pull stations`, `Manual call points`, `Fire alarm notification appliances`가 있음.
- 우리 클래스가 "화재경보기"라면 `Fire alarms`, `Fire alarm pull stations`, `Manual call points`가 가장 적합하다.

### Clap

- Clapping category: https://commons.wikimedia.org/wiki/Category:Clapping
- People clapping: https://commons.wikimedia.org/wiki/Category:People_clapping
- Applause: https://commons.wikimedia.org/wiki/Category:Applause

확인한 내용:

- `Clapping` 카테고리에는 `People clapping`, `Applause` 하위 카테고리가 있음.
- Commons 페이지 기준 `People clapping`은 71개 파일, `Applause`는 449개 파일이 있는 하위 카테고리로 표시됨.
- `clap` 클래스 보충에는 `People clapping`이 가장 적합하고, `Applause`는 군중/공연 장면이 섞일 수 있어 정제가 필요하다.

주의:

- Wikimedia Commons는 데이터셋이라기보다 공개 이미지 저장소다.
- 파일별 라이선스가 다르므로 공모전 제출/배포 전에 각 이미지 라이선스를 확인해야 한다.
- 사진 품질과 구도가 제각각이라 학습 전 수동 검수가 필요하다.

## 4순위: Kaggle

Kaggle은 검색 결과가 자주 바뀌고 데이터 출처가 불명확한 경우가 있어서 1순위로 추천하지 않는다.
그래도 추가 데이터가 필요하면 아래 검색어로 찾으면 된다.

- https://www.kaggle.com/datasets?search=doorbell
- https://www.kaggle.com/datasets?search=fire%20alarm
- https://www.kaggle.com/datasets?search=clapping
- https://www.kaggle.com/datasets?search=applause

추천 사용:

- Open Images만으로 이미지가 부족할 때 보충용
- 반드시 라이선스와 이미지 출처 확인

## 최종 추천

이번 요청이 "일반 이미지로 박수, 초인종, 화재경보기 데이터"라면 이렇게 가는 게 제일 깔끔하다.

1. **Open Images V7**에서 세 라벨을 가져오기
   - `Applause` -> `clap`
   - `Doorbell` -> `doorbell`
   - `Fire alarm system` -> `fire_alarm`

2. 부족한 클래스는 Wikimedia Commons에서 보충
   - `doorbell`: Doorbell buttons
   - `fire_alarm`: Fire alarms / Fire alarm pull stations / Manual call points
   - `clap`: People clapping

3. 객체 위치 박스가 필요하면 Roboflow에서 보충
   - 특히 `doorbell`, `fire_alarm`

4. 공모전 제출용이면 직접 찍은 이미지도 조금 섞기
   - 우리 주변 초인종
   - 학교/건물 비상벨
   - 실제 박수치는 사진

## 추천 수집 비율

처음 실험용:

| 클래스 | Open Images | Wikimedia/Roboflow/직접 촬영 | 총 목표 |
| --- | ---: | ---: | ---: |
| `clap` | 100장 | 50장 | 150장 |
| `doorbell` | 100장 | 50장 | 150장 |
| `fire_alarm` | 100장 | 50장 | 150장 |

공모전용 최소 권장:

| 클래스 | 목표 |
| --- | ---: |
| `clap` | 300장 이상 |
| `doorbell` | 300장 이상 |
| `fire_alarm` | 300장 이상 |

정제 기준:

- 너무 작은 물체만 있는 사진 제거
- 그림, 아이콘, 회로도, 안내판만 있는 이미지 제거
- `fire_alarm`은 실제 경보기/비상벨/수동 발신기 위주로 남기기
- `doorbell`은 문 옆 초인종 버튼 위주로 남기기
- `clap`은 손이 실제로 박수치는 장면 위주로 남기기

## 발표용 문장

일반 이미지 기반 분류 데이터는 Open Images V7에서 `Applause`, `Doorbell`, `Fire alarm system` 라벨을 확인하여 구성할 수 있다. 다만 해당 라벨들은 image-level classification 라벨이므로 객체 위치 탐지에는 적합하지 않고, YOLO 방식의 객체 탐지가 필요한 경우 Roboflow 또는 직접 라벨링을 병행해야 한다.
