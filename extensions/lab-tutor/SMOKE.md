# Lab tutor — Phase 1 smoke verification

Date: 2026-05-14
HEAD: 6f2e8a6c
Image tag: course-gen-learner-studio:67c2e199d8d3

## A. FastAPI tutor routes

- `/v1/tutor/chat` (valid request) → 200, `{"reply":"(stub) Got: hello","hint_tier":null}`
- `/v1/tutor/submit` (valid request) → 200, `{"test_results":{"passed":true,"details":"stub"},"viva_questions":[{"prompt":"Explain why you chose this data structure."},{"prompt":"Walk through your error handling."}]}`
- `/v1/tutor/chat` (empty session_id) → 422

PASS

## B. Docker image contains the extension

- Image tag built: yes (all layers cached from prior build)
- `/opt/lab-tutor/extensions/` contents: `extensions.json  scaler.lab-tutor-0.1.0/`
- `scaler.lab-tutor-0.1.0/` contents: `changelog.md  dist/  media/  package.json  readme.md`
- `dist/` contents: `extension.js`

PASS

## C. Container env vars + extension listing

- `code-server --list-extensions --show-versions` output: `scaler.lab-tutor@0.1.0`
- `env | grep LAB_TUTOR` output:
  ```
  LAB_TUTOR_BASE_URL=http://lab-tutor.svc:8000
  LAB_TUTOR_SESSION_ID=smoke_session_123
  ```

PASS

## D. Browser verification (manual — NOT done in this smoke)

Remaining manual checks for Tushar:
- Open `http://127.0.0.1:<host_port>/` in a browser after launching a learner workspace through the normal LMS flow.
- Verify the Lab Tutor icon appears in the activity bar.
- Click it; type "hello" in the sidebar; expect "(stub) Got: hello".
- Click the Submit button in the editor title bar; expect tests-passed toast + viva popup with two questions.

## Notes / deviations

- Server was started on port 8011 via uvicorn and responded immediately; killed cleanly via SIGTERM after route checks.
- Docker image was not pre-built; `docker build` completed in seconds as all layers were already cached.
- `code-server --list-extensions` is supported by the installed code-server version and prints `scaler.lab-tutor@0.1.0` exactly as expected.
- No deviations from the task spec required.
