# unaveraged — 개인정보처리방침 / Privacy Policy

최종 수정: 2026-10-29. 적용 대상: unaveraged Digit 및 unaveraged VQA Chrome·Firefox 확장 프로그램.

## 처리 목적과 범위

이 확장 프로그램은 smartstore.naver.com과 shopping.naver.com의 지원되는 영수증 CAPTCHA 페이지에서 이미지와 질문을 읽고 답변 칸을 채웁니다. 영수증 이미지, 질문, 편집 가능한 답변 칸이 보일 때 동작합니다. 기본값은 사용자가 입력된 답을 확인한 뒤 직접 제출하는 방식입니다. 설정에서 감시 URL과 요소 선택자를 변경하고 다른 사이트에 대한 접근을 허용할 수 있습니다. 이 확장은 NAVER 또는 Google이 제공하거나 승인한 제품이 아닙니다.

확장은 해당 페이지의 영수증 이미지 픽셀, 질문 문장, 답변 칸의 현재 값, 읽은 숫자 기록 또는 생성한 영수증 답변과 분류 정보를 기기에서 처리합니다. 영수증 이미지에는 상점 정보, 전화번호, 주소 또는 구매 내역이 포함될 수 있습니다. 현재 답변 값은 사용자가 작성 중인 내용을 덮어쓰지 않도록 확인합니다. 요청한 페이지의 URL은 지원 사이트인지 확인하는 데 사용합니다. 전체 방문 기록이나 다른 사이트의 페이지 내용을 수집하지 않습니다.

## 외부 전송과 대상 사이트

확장 자체는 영수증 이미지, 질문, 추론 결과 또는 사용 통계를 개발자나 외부 추론 서버로 전송하지 않습니다. 계정 가입, 분석 도구, 광고, 추적 기능은 없습니다. 모델과 JavaScript·WebAssembly 실행 코드는 설치 패키지 안에 포함되어 있으며, 실행 중 Hugging Face나 npm·PyPI에 연결하여 모델 또는 실행 코드를 받지 않습니다.

추론한 답변은 사용자가 보고 있는 대상 사이트의 입력칸에 쓰고 input/change 이벤트를 발생시킵니다. 따라서 그 사이트의 스크립트가 제출 전에도 입력값을 읽거나 자체적으로 전송할 수 있습니다. 대상 사이트의 데이터 처리는 해당 사이트의 개인정보처리방침을 따릅니다. 자동 제출은 기본적으로 꺼져 있습니다. 사용자가 설정에서 켜고 제출 버튼 선택자를 지정하면, 성공한 답변을 자동 입력한 뒤 해당 버튼을 클릭합니다. 이 동작은 대상 사이트의 폼 제출이나 다른 버튼 동작을 실행할 수 있습니다.

## 보관과 삭제

이미지와 질문은 처리 중 메모리에 존재합니다. 각 확장은 기본 1개, 설정에 따라 최대 4개의 추론 Worker를 사용하며 각 Worker는 반복 추론을 줄이기 위해 최대 32개 결과를 메모리에 보관합니다. unaveraged Digit는 이미지 해시와 숫자 기록을, unaveraged VQA는 이미지 해시·정규화한 질문·생성한 답변을 캐시하며 히트맵을 켜면 제한된 공간 어텐션 배열도 캐시하지만 원본 이미지 파일이나 KV 텐서를 결과 캐시에 저장하지 않습니다. 히트맵은 기본적으로 켜져 있으며, 팝업 미리보기를 위해 가장 최근 영수증 이미지 1장과 어텐션·답변을 별도 메모리에 보관합니다. 새 결과로 교체되고, 히트맵을 끄거나 추론 워커를 해제하면 삭제됩니다. 실제 읽기 작업이 끝난 뒤 설정한 유휴 시간(기본 약 1분) 동안 새 요청이 없으면 추론 Worker를 종료하여 모델·WASM 메모리·해당 캐시를 해제합니다. Chrome에서는 offscreen 문서도 닫습니다. Firefox는 비영구 백그라운드 페이지를 사용하며 브라우저가 유휴 페이지를 더 일찍 종료할 수도 있습니다. 상태 조회는 유휴 시간을 연장하지 않습니다. 확장 아이콘과 열린 지원 페이지의 감지 스크립트는 남으며, 다음 읽기 요청에서 모델을 다시 불러옵니다. 브라우저의 실행 상태에 따라 종료 및 메모리 회수 시점은 달라질 수 있습니다.

페이지에 표시한 결과, 현재 질문·이미지 참조, 입력한 답변은 해당 페이지가 열려 있는 동안 남을 수 있습니다. 페이지 이동·닫기 또는 영수증 변경에 따라 페이지 측 상태가 정리됩니다. 확장은 이러한 데이터를 파일, Chrome 동기화 저장소, localStorage 또는 IndexedDB에 기록하지 않습니다. 감시 URL·요소 선택자·워커 수·유휴 시간·자동 제출 등의 환경 설정은 브라우저의 확장 전용 로컬 저장소에 저장합니다. 프로파일링을 켜면 실행 시간·대기열·캐시·엔진 트레이스 등 제한된 진단 정보를 메모리에 보관할 수 있으며, 사용자가 요청하면 진단 JSON 파일을 기기에 내려받습니다. 진단 기록에 영수증 이미지·질문·답변을 포함하지 않습니다. 즉시 사용을 중단하려면 확장을 사용 중지하고 관련 탭을 닫으세요. 입력한 값에 대한 대상 사이트의 보관은 확장이 제어하지 않습니다.

## 권한과 제한된 사용

지원 사이트 접근은 해당 페이지의 이미지·질문을 읽고 답변을 입력하기 위해 사용합니다. Chrome의 offscreen 권한은 여러 탭이 설정한 기기 내 추론 Worker 풀을 공유하도록 사용합니다. Firefox에서는 지원 사이트 접근 권한과 백그라운드 페이지를 사용하며 offscreen 권한은 요청하지 않습니다. storage 권한은 환경 설정을 로컬에 저장하며, scripting 권한은 사용자가 추가한 감시 사이트에 감지 코드를 등록합니다. 추가 사이트 접근은 설정 저장 시 사용자가 허용한 경우에만 요청·사용합니다. 기본 localhost와 file URL 감시는 개발용 빌드에만 포함됩니다.

이 확장의 사용자 데이터 사용은 Chrome 웹 스토어 사용자 데이터 정책과 Limited Use 요구사항을 따릅니다. 데이터는 설명한 영수증 답변 입력 기능에만 사용합니다. 판매, 광고, 사용자 프로파일링, 신용 평가 또는 모델 학습에 사용하지 않으며 개발자에게 전송하거나 개발자가 열람하지 않습니다.

## 문의와 변경

문의는 Chrome 웹 스토어의 이 확장 프로그램 페이지에 표시된 개발자 연락처를 이용하세요. 소스 배포본은 배포된 저장소의 유지관리자에게 문의할 수 있습니다. 문의할 때 비공개 영수증 이미지나 인증 정보를 제공할 필요는 없습니다. 데이터 처리 방식이 바뀌면 이 문서와 스토어 설명을 갱신하고 필요한 안내를 제공합니다.

## English — purpose and data processed

unaveraged Digit and unaveraged VQA fill answers to supported receipt CAPTCHA questions on smartstore.naver.com and shopping.naver.com by default, or on sites explicitly configured and permitted by the user. It processes receipt image pixels, question text, the existing answer field, decoded digit records or generated answers and their field/category information on your device. Receipt images may contain store details, phone numbers, addresses, or purchase information. The page URL is checked against the supported sites. The extension does not collect your browsing history. It is independently developed and is not affiliated with NAVER or Google.

## English — transmission and retention

The extension itself does not send receipt images, questions, results, or usage statistics to the developer or a remote inference service. It contains no analytics, advertising, tracking, or account system. Code and model files are bundled with the installed extension; no runtime download from Hugging Face, npm, or PyPI is needed.

Answers are written into the target website's input field, with input/change events. The website can read or transmit that value even before you submit it, according to its own privacy practices. Automatic submission is off by default. If you enable it and configure a submit-button selector, the extension clicks that button after successfully filling an automatic answer; the target site may then submit its form or run the button action.

Images and questions are held in memory while processing. Each extension uses one inference worker by default, configurable up to four, with up to 32 cached results per worker. unaveraged Digit caches image hashes and digit records; unaveraged VQA caches image hashes, normalized questions, and generated answers. When heatmaps are enabled, bounded spatial attention arrays are cached as well. Original images and KV tensors are not kept in the result cache. Heatmaps are enabled by default. A separate in-memory preview holds only the most recent receipt image, attention and answer; it is replaced by a new result and cleared when heatmaps are disabled or workers are released. After the configured idle timeout (approximately one minute by default) without another read request following completed work, the worker terminates and releases its model, WASM memory and cache. Chrome also closes its offscreen document. Firefox uses a non-persistent background page that the browser may unload earlier when idle. Status checks do not extend the idle deadline. The toolbar button and detection scripts on open supported pages remain available; the next read reloads the model. Browser scheduling and memory allocation can affect when memory is reclaimed. Displayed results and page references can remain until the receipt changes or the page is navigated away from or closed. No receipt data is written by the extension to files, synchronized storage, localStorage, or IndexedDB. Preferences, including watched URLs, selectors, worker counts, idle timeout and automatic submission, are stored in extension-local browser storage. Optional profiling retains bounded execution metadata in memory; a user-requested diagnostic JSON export saves that metadata locally without receipt images, questions or answers. Disable the extension and close the relevant tabs to stop using it immediately. The extension cannot control retention by the target website.

## English — permissions, limited use, and contact

Access to the two supported sites is used to read the receipt and fill its answer. Chrome uses the offscreen permission to share the configured local inference workers across tabs. Firefox uses supported-site host permissions and a background page without requesting offscreen permission. The storage permission saves preferences locally; scripting registers detection scripts for additional watched sites. Access to additional sites is requested when the user saves settings and is used only after permission is granted. Default localhost and file URL watches are included only in development builds.

The extension's use of user data complies with the Chrome Web Store User Data Policy, including the Limited Use requirements. Data is used only for the disclosed receipt-answering feature, and is not sold, used for advertising, profiling, credit decisions, or model training, or transmitted to or read by the developer.

Contact the developer through the contact details on this extension's Chrome Web Store listing, or the maintainer of the repository distributing the source. You do not need to include private receipts or authentication information in a support request. Changes to data handling will be reflected in this policy and the store listing, with any required disclosures.
