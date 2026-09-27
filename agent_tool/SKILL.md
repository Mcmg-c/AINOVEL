# AINOVEL 小说全流程生成（Agent 工具）

让一个运行中的 AINOVEL 服务，根据参考文档 + 一句话需求，跑完整条 LangGraph 流水线并产出全书方案：
三套方案 → 三方辩论 → 主持人裁决 → 自动采用 → 逐事件拆解评分 → 合并路线（含子事件）→ 串行写出各章正文。

## 何时使用
- 用户/任务要求「根据文档生成世界观、大纲、多条路线或整本书」。
- 需要成书量级的大纲与正文，而不是一次性的闲聊结果。

## 前置条件
- AINOVEL 服务已在运行（默认 Linux 服务器，端口 8123）。
- 环境变量 `AINOVEL_URL`（默认 `http://127.0.0.1:8123`）与 `AINOVEL_TOKEN`（服务端设置了才需要）。

## 调用方式
优先使用仓库里的客户端：

```bash
python3 agent_tool/ainovel_client.py run \
  --project /opt/ainovel-projects/项目目录 \
  --guidance "都市悬疑，双男主，年下，破镜重圆" \
  --docs /home/user/docs/参考文档.txt \
  --words 90000 --chapters 36 --per-chapter 2500 \
  --timeout 1800
```

也可以直接 HTTP：

```bash
curl -X POST "${AINOVEL_URL:-http://127.0.0.1:8123}/api/agent/run" \
  -H "Content-Type: application/json" \
  -H "X-AINOVEL-Token: ${AINOVEL_TOKEN}" \
  -d '{"project":"/opt/ainovel-projects/项目目录","guidance":"都市悬疑，双男主","docs":["/home/user/docs/参考文档.txt"],"target_words":90000,"target_chapters":36,"words_per_chapter":2500,"workers":4,"wait":true,"timeout":1800}'
```

查询进度（配合 `wait:false`）：`GET /api/agent/status?path=<项目目录>`。

## 参数
- `project`（必填）：服务器本机上的项目目录，建议每个书名一个目录。
- `guidance`（必填）：一句话说清题材 / 标签 / 关系走向 / 想要的氛围。
- `docs`（可选）：服务器本机的 .txt / .md / .epub 文件路径列表；文件应先用工具放到服务器上。
- `target_words` / `target_chapters` / `words_per_chapter`：全书的体量规模（三项都要 > 0；也可预先在项目里保存）。
- `wait`（默认 true）：true 时 HTTP 阻塞到流程结束；false 立即返回 job，用 status 轮询。
- `timeout`（默认 1800 秒，最大 3600）；`workers`（并行度，默认 4）；`rounds`（攻辩轮数 1-3）。

## 返回结果
- `ok`：是否成功；失败看 `error`。
- `job`：流水线任务状态（queued / running / done / error），`stage` 为当前阶段，`detail` 为进度文案。
- `summary`：快速摘要（书名、合并路线名、总章数、已写完章数、规模）。
- `book`：完整项目数据。各章正文在
  `book.units[*].merged_route.chapters[*].draft`；
  章节目录与子事件在 `book.units[*].merged_route.chapters[*]`（`sub_events` 是本章包含的原子事件及各自字数）。

## 注意事项
- 全流程耗时跟文档长度、章节数、模型速度强相关，一般 5-20 分钟；
  调用方必须允许长超时，或者用 `wait:false` + `status` 轮询。
- 项目目录是服务器路径，不经过本地；启动前用 shell 把文档传过去。
- 失败重试安全：已生成的方案/路线/正文都会存盘，同目录重跑会继续推进。
