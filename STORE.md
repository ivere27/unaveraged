# Chrome 웹 스토어 등록

Digit와 VQA는 각각 별도의 스토어 항목으로 등록합니다. 코드 저장소와
개인정보처리방침 URL은 공유할 수 있습니다.

## 배포 파일

[설치 안내](README.md#install)의 일반 Chrome 배포 ZIP을 업로드하세요.
지원 브라우저 버전과 패키지 경로는 해당 안내를 기준으로 합니다. 각 ZIP 루트에는
`manifest.json`이 있으며, `chrome-dev`와 소스 ZIP은 스토어 설치 패키지로 사용하지 않습니다.
검사와 배포 명령은 [테스트](README.md#test)와 [릴리스 안내](README.md#github-releases)를 따릅니다.

## 홈페이지와 개인정보처리방침 URL

웹 문서 생성과 게시 방법은 [공개 문서 안내](README.md#privacy-and-public-documentation)를
따릅니다. 정책 원본은 [PRIVACY.md](PRIVACY.md)이며, 생성된 HTML을 별도로 편집하지 않습니다.
게시가 끝나면 스토어에 입력할 페이지가 로그인 없이 열리는지 확인하세요.

일반 GitHub Pages 프로젝트의 URL은 다음과 같습니다. 실제 소유자와 저장소명으로 바꾸세요.

| 용도 | URL |
| --- | --- |
| 홈페이지 | `https://<owner>.github.io/<repo>/` |
| 개인정보처리방침 | `https://<owner>.github.io/<repo>/privacy.html` |

두 스토어 항목의 개인정보처리방침 필드에 같은 정책 URL을 입력합니다.
도메인과 저장소명이 정해지기 전에는 위 예시를 실제 공개 URL로 사용하지 마세요.
개발자명, 문의 이메일, 홈페이지와 저장소 주소도 실제 배포 정보로 입력합니다.

## Digit 등록 문구

- **Item ID:** `aibjkkjaomgeppjjeekimpdoonndhmmb`
- **스토어 공개 URL:** `https://chromewebstore.google.com/detail/aibjkkjaomgeppjjeekimpdoonndhmmb`
- **개발자 콘솔 편집:** `https://chrome.google.com/webstore/devconsole/d818c2f3-ad14-4f9d-9c6e-cf99c8fa8dee/aibjkkjaomgeppjjeekimpdoonndhmmb/edit`
- **기본 언어:** 한국어 (ko)
- **카테고리:** 생산성 (Productivity) 또는 도구 (Tools)

**이름:** unaveraged Digit · 영수증 숫자 읽기

**한국어 짧은 설명:**

지원 사이트의 영수증 확인 질문에서 전화번호·주소 숫자를 기기에서 읽고 답변 칸을 채웁니다. 제출 전 답을 확인하세요.

**English short description:**

Read receipt digits on supported sites locally and fill the answer field. Review the answer before submitting.

**한국어 상세 설명:**

unaveraged Digit는 네이버 스마트스토어와 네이버 쇼핑 등의 영수증 확인 질문에서 전화번호 자리와 주소 숫자 질문의 답변 입력을 돕습니다. 한국어·영어 질문을 처리하고, 기기에서 읽은 숫자 기록과 답을 표시합니다. 답변 칸을 채운 뒤 기본값은 사용자가 직접 제출하는 방식입니다. 설정에서 제출 버튼을 지정하면 자동 제출을 켤 수 있습니다. 작성 중인 답을 보존하며 ‘다시 읽기’와 ‘답 입력’을 제공합니다. 상호명·상품·금액 질문이나 일반 문서 OCR은 지원하지 않습니다.

**English full description:**

unaveraged Digit helps fill answers to supported receipt verification questions on Naver Smart Store and Naver Shopping. It reads phone-number digits and numeric street addresses, handles Korean and English questions, and displays the digit record and answer. By default, you review and submit the filled answer. Automatic submission can be enabled for an explicitly configured button. Your manual edits are preserved. Store-name, item, price and general OCR questions are outside its scope.

**단일 목적 / Single purpose:**

지원되는 영수증 확인 화면의 전화번호·주소 숫자를 기기에서 읽고 해당 질문의 답을 입력합니다.
Read phone/address digits locally from supported receipt images and fill the answer for review.

## VQA 등록 문구

- **Item ID:** `fdbkjhljjjacepioaemjjigaicbammaa`
- **스토어 공개 URL:** `https://chromewebstore.google.com/detail/fdbkjhljjjacepioaemjjigaicbammaa`
- **개발자 콘솔 편집:** `https://chrome.google.com/webstore/devconsole/d818c2f3-ad14-4f9d-9c6e-cf99c8fa8dee/fdbkjhljjjacepioaemjjigaicbammaa/edit`
- **기본 언어:** 한국어 (ko)
- **카테고리:** 생산성 (Productivity) 또는 도구 (Tools)

**이름:** unaveraged VQA · 영수증 질문 답변

**한국어 짧은 설명:**

지원 사이트의 영수증 확인 화면에서 이미지와 질문을 기기에서 읽고 답변 칸을 채웁니다. 제출 전 답을 확인하세요.

**English short description:**

Answer supported receipt verification questions locally and fill the answer field. Review the answer before submitting.

**한국어 상세 설명:**

unaveraged VQA는 네이버 스마트스토어와 네이버 쇼핑 등의 영수증 확인 화면에서 이미지와 질문을 함께 읽고 답변 입력을 돕습니다. 한국어·영어로 전화번호·주소 숫자·상호명·상품명·수량·금액 등 학습된 질문을 처리합니다. 답을 생성하여 표시하고 답변 칸을 채우며, 기본값은 사용자가 직접 제출하는 방식입니다. 설정에서 제출 버튼을 지정하면 자동 제출을 켤 수 있습니다. 작성 중인 답을 보존합니다. 영수증 질문용 모델이며 일반 사진·문서의 범용 VQA 기능은 제공하지 않습니다.

**English full description:**

unaveraged VQA reads the image and question together to help fill supported receipt verification answers on Naver Smart Store and Naver Shopping. It handles trained Korean and English question types about stores, phone numbers, addresses, items, quantities and totals. It generates and displays an answer for you to review and submit by default, preserving your manual edits. Automatic submission is optional and requires a configured button selector. It is a receipt-domain model, not a general-purpose VQA system.

**단일 목적 / Single purpose:**

지원되는 영수증 확인 화면의 이미지와 질문을 기기에서 읽고 해당 답변을 입력합니다.
Read receipt images and questions locally on supported pages and fill the generated answer for review.

## 공통 설명과 권한

두 스토어 항목의 데이터 사용 신고와 공통 설명은
[개인정보처리방침](PRIVACY.md)의 한국어·영어 원문과 실제 동작을 기준으로 작성합니다.
기기에서만 처리하는 데이터도 처리 범위를 공개해야 합니다.

- 처리하는 데이터, 대상 사이트와 자동 제출: [처리 목적과 범위](PRIVACY.md#처리-목적과-범위),
  [외부 전송과 대상 사이트](PRIVACY.md#외부-전송과-대상-사이트).
- 메모리 보관, 유휴 해제, 설정 저장과 진단 내보내기: [보관과 삭제](PRIVACY.md#보관과-삭제).
- `storage`, `scripting`, `offscreen`, 사이트 접근 및 데이터 사용 제한:
  [권한과 제한된 사용](PRIVACY.md#권한과-제한된-사용).

[Chrome 사용자 데이터 안내](https://developer.chrome.com/docs/webstore/program-policies/user-data-faq)

## 제출 전 확인

1. 두 일반 Chrome 배포 ZIP에 대해 [테스트 절차](README.md#test)를 통과시킵니다.
2. 공개 합성 영수증으로 동작 화면을 준비합니다. 크롬 웹스토어 규격(1280x800, 24비트 PNG)에 맞춘 스크린샷이
   `docs/assets/screenshots/digit/` 및 `docs/assets/screenshots/vqa/`에 준비되어 있습니다. 데모임을 표시하고 비공개 영수증은 넣지 않습니다.
3. 설명·단일 목적·권한·데이터 사용·개인정보처리방침 URL·개발자 연락처를 입력합니다.
4. 지원 사이트에서 영수증 질문 확인 → 답변 직접 수정 → 다시 읽기 → 답 입력 → 직접 제출
   순서로 검토할 수 있도록 심사자 안내를 제공합니다.
5. 실제 사이트가 항상 영수증 CAPTCHA를 표시하지는 않으므로 요청받은 재현 절차를 제공합니다.

스토어 제출 시 실제 동작을 CAPTCHA 답변 입력 보조로 설명하고 현재 정책과 이미지 규격을 확인하세요.

[게시 절차](https://developer.chrome.com/docs/webstore/publish) ·
[개인정보 항목](https://developer.chrome.com/docs/webstore/cws-dashboard-privacy) ·
[이미지 규격](https://developer.chrome.com/docs/webstore/images)
