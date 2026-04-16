# DreamForge 商業化・產品化・國際化開發總計劃

## 0. 目標重定義（由研究 Demo 升級做產品）
你而家嘅新目標可以定義為：
1. **穩定性達到產品級**（可預期、可監控、可恢復）。
2. **敘事品質達到可交付級**（少雜訊、可控風格、可評分）。
3. **圖表達到專業分析級**（準確、可比較、可匯報）。
4. **價值表達清晰**（對用戶、研究者、商業客戶都講得明）。
5. **國際化 + 開源並行**（全球可用，同時保留 OSS 動能）。

---

## 1. 產品策略（Product Strategy）

## 1.1 產品定位
- **Core Positioning**：Computational Dream Intelligence Platform  
- **一句定位**：用生理模型 + 記憶圖譜 + LLM，將「夢境」變成可分析、可比較、可輸出嘅產品級數據與敘事。

## 1.2 目標客群（先後次序）
1. **B2B 研究/教育機構**（大學、lab、edtech）  
2. **B2B2C 內容與創作工具方**（遊戲、互動敘事、影像前期）  
3. **進階個人用戶**（自我追蹤、創作、sleep-tech 愛好者）

## 1.3 商業模式（保持開源）
- **Open-source Core + Commercial Cloud/Enterprise**（建議）
- 開源保留：
  - 核心模擬引擎、基礎 API、基本 dashboard
- 商業層可收費：
  - 團隊協作、權限/RBAC、審計、報告中心、批量作業、SLA、企業部署支援、進階模型包

---

## 2. 開發主軸（Workstreams）

## WS-A：平台穩定性（最高優先）
**目標**：將「可跑」升級到「可運營」。

### A1. SRE 基礎
- 定義 SLO：API 成功率、P95 latency、任務完成率、導出成功率。
- 統一 observability：structured logging、metrics、tracing、error taxonomy。
- 建立 incident runbook（LLM timeout、provider 400、export fail、queue backlog）。

### A2. 架構韌性
- 任務隊列化（長任務與 HTTP 脫鉤）。
- idempotency + retry policy + dead-letter。
- 快取與降級策略（LLM 不可用時仍可完成流程，但標記清晰）。

### A3. 品質閘（Quality Gate）
- CI 必過：lint/format/tests + 關鍵 smoke。
- release gate：穩定性 KPI 達標先發佈。

---

## WS-B：敘事品質（Narrative Excellence）
**目標**：由「能生成」提升到「可信、乾淨、可控」。

### B1. 內容潔淨層（已做方向，需產品化）
- 系統化 sanitization（prefix/no_think/html/token leakage）。
- sentence-safe trimming + style guardrail。
- 明確 fallback 標記與原因分類（對內可診斷，對外可解釋）。

### B2. 風格控制與評分
- 加入可配置 style preset（scientific / cinematic / minimal / therapeutic tone）。
- 內建品質評分：
  - coherence、novelty、memory-grounding、length compliance、artifact score
- 建立 Golden Set 回歸測試（固定 seed + 預期質量門檻）。

### B3. PromptOps / ModelOps
- prompt versioning + A/B。
- provider capability matrix（邊個 model 支持咩 payload）。
- 自動選模策略（成本、延遲、質量三者平衡）。

---

## WS-C：圖表專業性（Analytics-grade Visualization）
**目標**：由「好睇」變「可分析、可決策」。

### C1. 視覺規範
- 建立 chart design system（色板、字級、對比、annotation 規範）。
- 統一單位/軸/圖例/小數位規格。

### C2. 專業分析功能
- 加入 compare mode（run vs run、baseline vs counterfactual）。
- confidence band / anomaly flags / event markers（lucid events, replay peaks）。
- 報告模板輸出（PDF/PNG bundle + methodology appendix）。

### C3. 數據可信度
- 每張圖都有 data provenance（來源欄位 + 版本）。
- 可追溯到 segment-level raw record。

---

## WS-D：產品化工程（Productization）
**目標**：從 repo 工程變成可交付產品。

### D1. 產品邊界
- Edition 定義：Community / Pro / Enterprise（功能矩陣）。
- API versioning（v1/v2）、deprecation policy。

### D2. 安全與合規
- secrets 管理、audit log、RBAC、rate limit、abuse guard。
- 條款與聲明：非醫療聲明、資料使用聲明、模型限制透明化。

### D3. 開發流程
- roadmap-driven issue system（Epic → Milestone → Ticket）。
- PR template 強制：problem / design / risks / validation。

---

## WS-E：國際化（Internationalization）
**目標**：全球可讀、可用、可營運。

### E1. i18n/l10n
- UI 文案抽離（en 作主語言，zh-HK/zh-CN/ja 後續）。
- 時區、數值格式、日期格式、單位在地化。

### E2. 國際文件系統
- README、Docs、API docs、Quickstart 全面英文優先。
- 多語落地頁（產品價值 + 用例 + 限制聲明）。

### E3. 全球部署能力
- region-ready 配置（latency-aware endpoint、CDN 文檔）。
- 可選多模型供應商策略（OpenAI/Anthropic/Ollama/本地）。

---

## WS-F：開源增長與商業轉化（OSS + Revenue）
**目標**：維持開源勢能，同步建立可持續收入。

### F1. GitHub 增長引擎
- 「一鍵跑」體驗（Quickstart < 10 min）。
- 高質 demo 資產（短片、GIF、對比案例）。
- 每個 release 有可感知價值（穩定性、品質、圖表、報告）。

### F2. 商業漏斗
- 開源版導向：Newsletter / waitlist / Pro trial。
- 企業線索入口：`/enterprise`、技術白皮書、SLA sheet。

### F3. 社群治理
- RFC 機制、貢獻者等級、公開 roadmap。
- 避免只靠單人維護，建立 maintainers 結構。

---

## 3. 分階段執行計劃（無時間版）

## Phase 1：產品基線（Foundation）
**完成標準**
- SLO/監控/錯誤分類上線
- 內容 sanitization 與 narrative quality gate 上線
- chart design system v1
- API/version policy 定稿

## Phase 2：品質放大（Quality Scale）
**完成標準**
- 敘事評分與 A/B 框架可用
- compare mode + 報告模板可用
- 國際化框架（en-first + i18n key）完成

## Phase 3：商業準備（Commercial Readiness）
**完成標準**
- Community/Pro 功能邊界清晰
- 企業安全能力（RBAC/audit/rate-limit）可交付
- Pro 試用漏斗 + 企業資料頁上線

## Phase 4：國際化擴張（Global Expansion）
**完成標準**
- 多語介面與文件發布
- 多地部署與供應商策略穩定
- 形成可持續 release cadence 與社群節奏

---

## 4. KPI/衡量框架（建議）

## 技術 KPI
- API success rate、P95 latency、simulation completion rate
- narrative artifact rate（prefix/noise/html leakage）
- fallback ratio、provider error ratio

## 產品 KPI
- 報告導出成功率、compare 功能使用率
- 回訪率、活躍專案數、團隊協作啟用率

## 開源/商業 KPI
- star growth、active contributors、issue response time
- trial-to-paid conversion、enterprise pipeline 數量

---

## 5. 你而家可以即刻落地嘅首批任務（Top 12）
1. 定 SLO + error taxonomy  
2. 建 Prometheus/Grafana + tracing baseline  
3. 任務隊列化（長任務脫 HTTP）  
4. narrative quality scoring v1  
5. prompt/model version registry  
6. chart design system v1（含 export spec）  
7. compare/report 中心 v1  
8. API v1 contract freeze + changelog discipline  
9. RBAC/audit/rate-limit baseline  
10. en-first docs + i18n key 抽離  
11. OSS roadmap + RFC 流程上線  
12. Pro/Enterprise 功能矩陣與 pricing hypothesis

---

## 6. 最終建議（管理層視角）
你依家最啱嘅路線係：  
**先用「穩定性 + 品質 + 可分析性」建立產品可信度，再用「開源分發 + 商業能力分層」放大增長。**  

咁樣可以同時守住科研/開源公信力，同埋打開真正商業化空間。
