# DreamForge 近100工作流衝刺・修復與產品化執行總計劃

## 1) 今次重點問題與根因（已定位）

1. `localhost:8501` 以 `<18:00` 入睡會報錯  
   - 根因：API `SimulationConfig.sleep_start_hour` 限制 `ge=18.0`，導致午睡/清晨入睡被拒。
2. Dashboard「Test Connection」唔係真連線測試  
   - 根因：舊流程用本地 `LLMBackend.generate()`，可能只係 fallback，無保證同 API 後端 provider 真正連通。
3. Dashboard Status 顯示 `Detected: dreamscript` 與實際 LM Studio 連線不一致  
   - 根因：狀態來源係 dashboard 本地 auto-detect，唔係 API 實際配置與健康狀態。
4. 請求 timeout 默認偏低  
   - 根因：`SIM_REQUEST_TIMEOUT_SECONDS` 預設與 compose 設定仍未統一到 3600。
5. Simulation parameters/pharmacology 傳參不一致  
   - 根因：舊 payload 使用 `pharmacology.ssri_factor` 巢狀結構，API實際期待 `ssri_strength` 等頂層欄位。

---

## 2) 已執行修復與改進（落地項）

### A. 穩定性與參數兼容
- API `sleep_start_hour` 已放寬為 `0.0–26.0`，支援午睡與清晨入睡。
- 新增/對齊 pharmacology 欄位：`melatonin`、`cannabis`（同 `ssri_strength` 一齊入 payload/回傳配置）。
- Simulation summary 加入 `pharmacology_profile`，方便報告與排錯。

### B. Dashboard 正式連線產品化
- 「Test Connection」改為正式流程：
  1. `POST /api/llm/config` 套用當前 provider/base_url/model/timeout  
  2. `GET /api/health/llm` 驗證真實連線  
  3. 顯示可用模型列表/錯誤訊息
- 狀態顯示改為 API 真實健康狀態，而非本地 fallback detect。
- Provider UI 改為產品向選項（LM Studio / Ollama / OpenAI），避免 demo 誤導。

### C. Timeout 與體驗
- Runtime default `simulation_request_timeout_seconds` 改為 `3600`。
- `docker-compose.yml` dashboard `SIM_REQUEST_TIMEOUT_SECONDS` 改為 `"3600"`。
- Dashboard timeout 上限提升，避免長夜模擬被前端過早中止。

### D. Dashboard 參數與 UX
- Simulation payload 對齊 API：`duration_hours`, `dt_minutes`, `sleep_start_hour`, `stress_level`, `ssri_strength`, `melatonin`, `cannabis`, `emotional_state`, `use_llm`, `prior_day_events`。
- 「Sleep Start」改為支援 `0–26` 時段，文案清晰標示可午睡/清晨入睡。
- UI 視覺升級（漸層背景、玻璃卡片、按鈕交互細節），減少傳統純開發面板感。

---

## 3) 組合測試策略（simulation parameters / pharmacology / prior day events）

## 3.1 自動化回歸組合（已納入測試）
- 早晨入睡（`sleep_start_hour=5.5`）+ melatonin + 多事件。
- 午睡入睡（`sleep_start_hour=13.0`）+ cannabis + 高 stress。
- 深夜入睡（`sleep_start_hour=23.5`）+ 輕 stress + mixed pharmacology。

每組驗證：
- API `201` 成功回應。
- segments 非空。
- config 回傳值與輸入一致（包括 `melatonin`/`cannabis`）。
- summary 含 `pharmacology_profile` 並反映 `ssri_strength`。

## 3.2 連線測試驗證（已納入流程）
- LLM 設定更新成功。
- LLM health endpoint 可回覆，並返回模型清單（若 provider 支援）。
- 錯誤情況會展示可讀訊息，而非沉默 fallback。

---

## 4) 各工作流衝刺至接近100%策略（下一步執行藍圖）

## WS-A 平台穩定性（目標 95–100）
1. 補 tracing（OpenTelemetry spans：simulate/job/llm/export）。
2. 加 incident runbook（timeout/provider400/export fail/queue backlog）。
3. release gate 直接用 `/api/slo` 指標自動判斷。

## WS-B 敘事品質（目標 95–100）
1. 加 style presets（scientific/cinematic/minimal/therapeutic）。
2. Golden set + seed gate（quality 門檻阻擋回歸）。
3. Prompt profile A/B 與質量分數儀表板。

## WS-C 圖表專業性（目標 95–100）
1. compare 視圖加 confidence band / anomaly flags / event markers。
2. 匯出報告升級到 JSON + PNG/SVG bundle + methodology appendix。
3. chart-level provenance 加版本/來源欄位索引。

## WS-D 產品化工程（目標 95–100）
1. RBAC（viewer/editor/admin）與 token scope。
2. Audit 查詢端點與操作審計檢索。
3. PR template 與 roadmap-issue workflow 強制化。

## WS-E 國際化（目標 95–100）
1. dashboard 全 key 化（目前為基線，擴到完整 UI）。
2. 多語 docs（en 主、zh-HK/zh-CN 次）。
3. locale-aware 日期/數值/單位。

## WS-F OSS + 商業轉化（目標 95–100）
1. `/enterprise` landing + SLA sheet + waitlist/trial 流程。
2. release narrative（每版可感知價值）與 demo assets pipeline。
3. maintainer governance + RFC cadence。

---

## 5) 驗收標準（Definition of Done）

1. 所有核心 API 與 dashboard 參數路徑都有自動化測試覆蓋。  
2. 連線狀態、錯誤碼、回退原因可觀測且可解釋。  
3. 午睡/清晨入睡場景穩定可用，無 validation regression。  
4. 匯出/比較/報告鏈路穩定，無靜默失敗。  
5. UI/UX 達到演示級（互動、視覺、資訊層次）而唔係純工程面板。  
