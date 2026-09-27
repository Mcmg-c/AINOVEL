<script setup>
import { computed, markRaw, nextTick, onMounted, onUnmounted, reactive, ref } from 'vue'

const worldFields = [
  { key: 'premise', label: '世界前提', placeholder: '这个世界的核心设定' },
  { key: 'era_place', label: '时代与地点', placeholder: '例如：当代，南方老城区' },
  { key: 'rules', label: '规则与限制', placeholder: '什么是可能的、什么不可能' },
  { key: 'tone', label: '基调', placeholder: '例如：克制、温情、轻悬疑' },
]
const genrePresets = ['异世界', '都市', '修仙', '科幻', '奇幻', '悬疑', '历史', '校园', '末世', '武侠']

const form = reactive({ path: '', guidance: '' })
const book = reactive({ world: {}, characters: [], units: [], current: '', graph: { nodes: [], edges: [], events: [], facts: [] }, title: '', mainline: '', references: [], reference_limit: 0, target_words: 0, target_chapters: 0, words_per_chapter: 0 })
const charactersDraft = ref([])
const referencePlan = ref('')
const referenceBrief = ref(null)
const proposals = ref([])
const debate = ref(null)
const pipelineJob = reactive({ active: false, stage: '', stage_label: '', detail: '', done: 0, total: 0 })
let pipelinePollTimer = null
const pipelineNodes = [
  { key: 'generate', label: '并行生成三套方案', hint: '3 次独立 LLM 调用：详细主线 / 世界观 / 人物 / 大事件目录' },
  { key: 'advocate', label: '三个 Agent 立论', hint: '每个方案一个辩护 Agent，陈述核心优势与风险自检' },
  { key: 'rebuttal', label: '三个 Agent 攻辩', hint: '互相质疑硬伤，并回应别人对自己的质疑' },
  { key: 'moderator', label: '主持人裁决', hint: '评分排名；必要时融合出一套新方案' },
  { key: 'apply', label: '自动采用获胜方案', hint: '裁决结果直接写入：书名、主线、世界观、人物、大事件目录' },
  { key: 'outline', label: '逐事件拆解评分', hint: '每个事件独立调用：生成 A/B/C 三套章节拆分并打分，自动选最高分组成路线' },
  { key: 'merge', label: '校验合并路线', hint: '按章节数/每章字数收拢子事件：一章可包含多个子事件，逐章保留子事件明细' },
  { key: 'write', label: '串行撰写各章正文', hint: '逐章写正文，串行执行：每一章带上前文完整章节，并逐条覆盖本章子事件' },
]
const pipelineStageOrder = ['generate', 'advocate', 'rebuttal', 'moderator', 'apply', 'outline', 'merge', 'write', 'done']
const draftInstructions = reactive({})

const busy = ref('')
const notice = ref('')
const error = ref('')

const showTrace = ref(false)
const traceEvents = ref([])
const traceOffset = ref(0)
const traceNow = ref(Date.now())
const traceBox = ref(null)
const traceAutoScroll = ref(true)
let traceTimer = null
let clockTimer = null
const traceActive = ref([])
const expandedTraces = ref(new Set())
let traceSeq = 0

const showSettings = ref(false)
const config = ref({ llm_configs: {}, choose_configs: {}, other_params: {} })
const selectedPreset = ref('')
const draftPreset = reactive({ api_key: '', base_url: '', interface_format: 'OpenAI', model_name: '', temperature: 0.7, max_tokens: 8192, timeout: 600 })

const currentUnit = computed(() => book.units.find(u => u.id === book.current) || book.units[0] || null)
const ask = reactive({ show: false, text: '', scope: '', action: 'answer', question: '', input: '', result: '', busy: false, messages: [], usedRewrite: false, pendingRewrite: null, showRewriteConfirm: false, pendingRewriteError: '' })
const sel = reactive({ text: '', scope: '', full: '', ref: null, target: null, targets: [], range: null, start: 0, end: 0, x: 0, y: 0, visible: false })
const promptPreview = reactive({ show: false, title: '', kind: '', label: '', request: '', prompt: '', onSend: null, sending: false, error: '' })
const chatLogs = ref({})
const canApplyAsk = computed(() => !!sel.target || !!sel.targets.length || (!!sel.ref && typeof sel.ref.value === 'string'))

const api = async (url, options = {}) => {
  const isForm = typeof FormData !== 'undefined' && options.body instanceof FormData
  const headers = isForm ? { ...(options.headers || {}) } : { 'Content-Type': 'application/json', ...(options.headers || {}) }
  const response = await fetch(url, { headers, ...options })
  if (!response.ok) {
    let message = `请求失败 (${response.status})`
    try { message = (await response.json()).detail || message } catch {}
    throw new Error(message)
  }
  return response.json()
}

function clearMessage() { notice.value = ''; error.value = '' }
function applyBook(result) {
  const data = result?.book || result || {}
  Object.assign(book, data)
  book.title = book.title || ''
  book.mainline = book.mainline || ''
  book.target_words = Number(book.target_words) || 0
  book.target_chapters = Number(book.target_chapters) || 0
  book.words_per_chapter = Number(book.words_per_chapter) || 0
  book.references = book.references || []
  book.reference_limit = Number(book.reference_limit) || 0
  book.graph = book.graph || { nodes: [], edges: [], events: [], facts: [] }
  if (Array.isArray(data.proposals) && data.proposals.length) proposals.value = data.proposals
  if (data.debate && typeof data.debate === 'object') debate.value = data.debate
  syncCharacters()
}
function syncCharacters() {
  charactersDraft.value = (book.characters || []).map(item => ({ ...item }))
}

const TRACE_TYPE_LABEL = { request: '发出请求', attempt: '等待响应', response: '收到响应', tool_result: '工具返回', retry: '准备重试', error: '出错', failed: '失败' }
const traceStatus = computed(() => {
  if (!traceActive.value.length) return '当前没有进行中的模型请求'
  const state = traceActive.value[traceActive.value.length - 1]
  const startedAt = Date.parse((state.started_at || '').replace(' ', 'T')) || traceNow.value
  const seconds = Math.max(0, Math.round((traceNow.value - startedAt) / 1000))
  return `${state.label} · ${state.model} · 已等待 ${seconds} 秒`
})

async function pollTrace() {
  try {
    const [data, activeData] = await Promise.all([
      api(`/api/logs/llm/tail?offset=${traceOffset.value}`),
      api('/api/logs/llm/active'),
    ])
    traceActive.value = activeData.active || []
    if (data.events?.length) {
      const appended = data.events.map(event => ({ ...event, _uid: ++traceSeq }))
      traceEvents.value = [...traceEvents.value, ...appended].slice(-400)
      if (traceAutoScroll.value) {
        await nextTick()
        if (traceBox.value) traceBox.value.scrollTop = traceBox.value.scrollHeight
      }
    }
    traceOffset.value = data.offset
  } catch { /* 日志接口不可用时静默 */ }
}
function clearTrace() { traceEvents.value = []; expandedTraces.value = new Set() }
function toggleTrace(uid) {
  const next = new Set(expandedTraces.value)
  if (next.has(uid)) next.delete(uid)
  else next.add(uid)
  expandedTraces.value = next
}
function tracePreview(event) {
  const text = (event.content || '').replace(/\s+/g, ' ').trim()
  return text.length > 160 ? text.slice(0, 160) + '…' : text
}
async function openTrace() {
  showTrace.value = true
  traceOffset.value = 0
  traceEvents.value = []
  await pollTrace()
  if (!traceTimer) traceTimer = setInterval(pollTrace, 1000)
  if (!clockTimer) clockTimer = setInterval(() => { traceNow.value = Date.now() }, 1000)
}
function closeTrace() {
  showTrace.value = false
  clearInterval(traceTimer); traceTimer = null
  clearInterval(clockTimer); clockTimer = null
}
function finishTraceIfIdle() {
  setTimeout(() => { if (!traceActive.value.length) closeTrace() }, 6000)
}

async function runTask(task, busyText, doneText) {
  clearMessage(); busy.value = busyText
  try {
    const result = await task()
    applyBook(result)
    notice.value = doneText
  } catch (e) { error.value = e.message }
  finally { busy.value = ''; finishTraceIfIdle() }
}

async function runTaskStrict(task, busyText, doneText) {
  clearMessage(); busy.value = busyText
  try {
    const result = await task()
    applyBook(result)
    notice.value = doneText
    return result
  } catch (e) { error.value = e.message; throw e }
  finally { busy.value = ''; finishTraceIfIdle() }
}

async function requestPromptPreview(url, payload) {
  const data = await api(url, { method: 'POST', body: JSON.stringify({ ...payload, preview: true }) })
  return data.preview || {}
}
function openPromptPreview({ title, kind, label, request, prompt, onSend }) {
  promptPreview.title = title
  promptPreview.kind = kind || ''
  promptPreview.label = label || ''
  promptPreview.request = request || ''
  promptPreview.prompt = prompt || ''
  promptPreview.onSend = onSend || null
  promptPreview.error = ''
  promptPreview.sending = false
  promptPreview.show = true
}
function closePromptPreview() {
  if (promptPreview.sending) return
  promptPreview.show = false
  promptPreview.onSend = null
}
async function confirmPromptPreview() {
  if (!promptPreview.onSend || promptPreview.sending) return
  promptPreview.error = ''
  promptPreview.sending = true
  try {
    await promptPreview.onSend()
    promptPreview.show = false
    promptPreview.onSend = null
  } catch (e) {
    promptPreview.error = e.message || String(e)
  } finally {
    promptPreview.sending = false
  }
}

async function loadProject() {
  clearMessage()
  if (!form.path.trim()) { error.value = '请先填写项目保存路径'; return }
  busy.value = '正在读取项目'
  try {
    await persistForm()
    restoreChatLogs()
    applyBook(await api(`/api/book?path=${encodeURIComponent(form.path)}`))
    notice.value = '项目已加载'
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

async function persistForm() {
  config.value.other_params = { ...(config.value.other_params || {}), filepath: form.path }
  try { await api('/api/config', { method: 'PUT', body: JSON.stringify(config.value) }) } catch { /* 保存路径失败不打断使用 */ }
}

async function saveWorld() {
  if (!ensurePath()) return
  clearMessage(); busy.value = '正在保存世界观'
  try {
    applyBook(await api('/api/world', { method: 'PUT', body: JSON.stringify({ path: form.path, world: { ...book.world } }) }))
    notice.value = '世界观已保存'
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

async function saveCharacters() {
  if (!ensurePath()) return
  clearMessage(); busy.value = '正在保存人物'
  try {
    applyBook(await api('/api/characters', { method: 'PUT', body: JSON.stringify({ path: form.path, characters: charactersDraft.value }) }))
    notice.value = '人物已保存'
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

async function saveBookMeta() {
  const words = Number(book.target_words) || 0
  const chapters = Number(book.target_chapters) || 0
  const perChapter = Number(book.words_per_chapter) || 0
  if (words <= 0) { clearMessage(); error.value = '请先填写大概总字数（必填）'; return }
  if (chapters <= 0) { clearMessage(); error.value = '请填写大概章节数（必填），AI 会按这个规模合并路线，避免拆得太碎'; return }
  if (perChapter <= 0) { clearMessage(); error.value = '请填写每章大概字数（必填），AI 会按这个篇幅分配每章字数'; return }
  if (!ensurePath()) return
  clearMessage(); busy.value = '正在保存全书规模'
  try {
    applyBook(await api('/api/book/meta', { method: 'PUT', body: JSON.stringify({ path: form.path, title: book.title, mainline: book.mainline, target_words: words, target_chapters: chapters, words_per_chapter: perChapter }) }))
    notice.value = `全书规模已保存：约 ${words} 字 / ${chapters} 章 / 每章约 ${perChapter} 字`
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

function bookScaleReady() {
  return Number(book.target_words) > 0 && Number(book.target_chapters) > 0 && Number(book.words_per_chapter) > 0
}

function formatBytes(bytes) {
  const value = Number(bytes) || 0
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}
function uploadReference(event) {
  const file = event.target?.files?.[0]
  if (event.target) event.target.value = ''
  if (!ensurePath()) return
  if (!file) return
  if (!/\.(txt|md|epub)$/i.test(file.name || '')) { error.value = '只支持 .txt / .md / .epub 文件'; return }
  clearMessage(); busy.value = '正在上传并解析参考文本…'
  const body = new FormData()
  body.append('path', form.path)
  body.append('file', file)
  api('/api/references/upload', { method: 'POST', body })
    .then(data => {
      book.references = data.references || []
      notice.value = /\.epub$/i.test(file.name) ? 'epub 已转换为纯文本并导入' : '参考文本已上传'
    })
    .catch(e => { error.value = e.message })
    .finally(() => { busy.value = '' })
}
async function toggleReference(name, enabled) {
  if (!ensurePath()) return
  try {
    const data = await api(`/api/references/${encodeURIComponent(name)}`, { method: 'PUT', body: JSON.stringify({ path: form.path, enabled }) })
    book.references = data.references || []
  } catch (e) { error.value = e.message }
}
async function deleteReference(name) {
  if (!ensurePath()) return
  if (!window.confirm(`确定删除参考文本「${name}」吗？此操作不可恢复。`)) return
  try {
    const data = await api(`/api/references/${encodeURIComponent(name)}?path=${encodeURIComponent(form.path)}`, { method: 'DELETE' })
    book.references = data.references || []
    notice.value = '参考文本已删除'
  } catch (e) { error.value = e.message }
}
async function saveReferenceLimit() {
  if (!ensurePath()) return
  try {
    const data = await api('/api/reference-limit', { method: 'PUT', body: JSON.stringify({ path: form.path, limit: book.reference_limit }) })
    book.reference_limit = data.reference_limit
    notice.value = book.reference_limit === 0 ? '参考文本将以全文发送给 AI' : `参考文本将以节选（${book.reference_limit} 字符）发送给 AI`
  } catch (e) { error.value = e.message }
}
function pipelineNodeState(key) {
  if (pipelineJob.stage === 'done') return 'done'
  if (!pipelineJob.active && pipelineJob.stage !== 'done') return 'pending'
  const current = pipelineStageOrder.indexOf(pipelineJob.stage || 'queued')
  const mine = pipelineStageOrder.indexOf(key)
  if (current < 0) return 'pending'
  if (mine < current) return 'done'
  if (mine === current) return 'active'
  return 'pending'
}

function stopPipelinePoll() {
  if (pipelinePollTimer) { clearInterval(pipelinePollTimer); pipelinePollTimer = null }
  pipelineJob.active = false
}

async function startPipeline() {
  if (!ensurePath()) return
  if (!bookScaleReady()) { error.value = '请先保存全书规模（总字数、大概章节数、每章大概字数）再启动全流程'; return }
  if (!referencePlan.value.trim()) { error.value = '请先写一句想写什么（题材 / 标签 / 关系走向）'; return }
  clearMessage(); openTrace()
  try {
    const data = await api('/api/proposals/pipeline', { method: 'POST', body: JSON.stringify({ path: form.path, guidance: referencePlan.value, target_words: Number(book.target_words) || 0, target_chapters: Number(book.target_chapters) || 0, words_per_chapter: Number(book.words_per_chapter) || 0 }) })
    stopPipelinePoll()
    pipelineJob.active = true
    Object.assign(pipelineJob, data.job || {})
    notice.value = 'LangGraph 全流程已启动：生成三套方案 → 三方辩论 → 主持人裁决 → 自动采用 → 逐事件拆解评分 → 合并起名 → 串行撰写各章正文'
    pipelinePollTimer = setInterval(pollPipelineJob, 2000)
  } catch (e) { error.value = e.message }
}

async function pollPipelineJob() {
  try {
    const data = await api(`/api/proposals/pipeline?path=${encodeURIComponent(form.path)}`)
    const job = data.job || {}
    Object.assign(pipelineJob, job)
    pipelineJob.active = ['queued', 'running'].includes(job.status)
    if (job.status === 'error') {
      stopPipelinePoll()
      error.value = job.message || job.detail || '流水线失败'
      finishTraceIfIdle()
    } else if (job.status === 'done') {
      stopPipelinePoll()
      await applyBook(await api(`/api/book?path=${encodeURIComponent(form.path)}`))
      notice.value = 'LangGraph 全流程完成：辩论裁决 → 自动采用 → 逐事件拆解评分 → 合并起名 → 各章正文已串行写完'
      finishTraceIfIdle()
      goToUnits()
    }
  } catch (e) {
    stopPipelinePoll()
    error.value = e.message
    finishTraceIfIdle()
  }
}

async function generateReferenceBrief() {
  if (!ensurePath()) return
  clearMessage(); busy.value = '正在准备请求预览…'
  try {
    const payload = { path: form.path, guidance: referencePlan.value, target_words: Number(book.target_words) || 0 }
    const previewData = await requestPromptPreview('/api/references/brief', payload)
    busy.value = ''
    openPromptPreview({
      title: '根据参考文本生成设定',
      kind: previewData.kind || 'reference-brief',
      label: previewData.label || '参考文本 → 单主线 / 世界观 / 人物建议',
      request: previewData.request || '',
      prompt: previewData.prompt || '',
      onSend: async () => {
        busy.value = '正在分析参考文本'
        try {
          const data = await api('/api/references/brief', { method: 'POST', body: JSON.stringify(payload) })
          referenceBrief.value = data.brief || {}
          notice.value = '设定建议已生成，检查后可一键应用到本书'
        } catch (e) { error.value = e.message; throw e }
        finally { busy.value = '' }
      },
    })
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function applyReferenceBrief() {
  const brief = referenceBrief.value
  if (!brief || !ensurePath()) return
  if (brief.title) book.title = brief.title
  if (brief.mainline) book.mainline = brief.mainline
  const world = brief.world || {}
  for (const key of ['genre', 'premise', 'era_place', 'rules', 'tone']) {
    if (world[key]) book.world[key] = world[key]
  }
  if (Array.isArray(brief.characters) && brief.characters.length) {
    charactersDraft.value = brief.characters.map(item => ({ name: item.name || '', gender: item.gender || '', age: item.age || '', notes: item.notes || '' }))
  }
  clearMessage(); busy.value = '正在应用设定并保存…'
  try {
    await api('/api/book/meta', { method: 'PUT', body: JSON.stringify({ path: form.path, title: book.title, mainline: book.mainline }) })
    await api('/api/world', { method: 'PUT', body: JSON.stringify({ path: form.path, world: { ...book.world } }) })
    const result = await api('/api/characters', { method: 'PUT', body: JSON.stringify({ path: form.path, characters: charactersDraft.value }) })
    applyBook(result)
    notice.value = '已保存：单主线、世界观与人物均已应用'
    nextTick(() => document.querySelector('.setup-page')?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

function ensurePath() {
  if (!form.path.trim()) { error.value = '请先填写项目保存路径'; return false }
  return true
}

function addCharacter() {
  charactersDraft.value.push({ name: '', gender: '', age: '', notes: '' })
}
function removeCharacter(index) { charactersDraft.value.splice(index, 1) }

async function createUnit() {
  if (!ensurePath()) return
  try {
    applyBook(await api('/api/units', { method: 'POST', body: JSON.stringify({ path: form.path, title: '', goal: '', cast: [], lead_in: '', word_count: 1500 }) }))
    notice.value = '已初始化全书工作区，可以直接让 AI 出大事件目录'
    goToUnits()
  } catch (e) { error.value = e.message }
}

function goToUnits() {
  document.getElementById('units-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function goToProposals() {
  document.getElementById('proposals-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

async function selectUnit(id) {
  try {
    applyBook(await api(`/api/units/${id}/select`, { method: 'POST', body: JSON.stringify({ path: form.path }) }))
  } catch (e) { error.value = e.message }
}

async function deleteUnit(id) {
  if (!window.confirm('确定删除这个章节吗？此操作不可恢复。')) return
  try {
    applyBook(await api(`/api/units/${id}?path=${encodeURIComponent(form.path)}`, { method: 'DELETE' }))
    notice.value = '章节已删除'
  } catch (e) { error.value = e.message }
}

async function persistUnit() {
  const unit = currentUnit.value
  if (!unit) return
  const result = await api(`/api/units/${unit.id}`, { method: 'PUT', body: JSON.stringify({ path: form.path, title: unit.title || '', goal: unit.goal || '', cast: unit.cast || [], lead_in: unit.lead_in || '', word_count: unit.word_count || 1500 }) })
  applyBook(result)
  const metaResult = await api('/api/book/meta', { method: 'PUT', body: JSON.stringify({ path: form.path, title: book.title, mainline: book.mainline, target_words: Number(book.target_words) || 0, target_chapters: Number(book.target_chapters) || 0, words_per_chapter: Number(book.words_per_chapter) || 0 }) })
  applyBook(metaResult)
}
async function saveEvent(eventSeq) {
  const unit = currentUnit.value
  if (!ensurePath() || !unit) return
  const event = (unit.segments || []).find(item => item.seq === eventSeq)
  if (!event) return
  clearMessage(); busy.value = '正在保存事件'
  try {
    applyBook(await api(`/api/units/${unit.id}/segments/${eventSeq}`, { method: 'PUT', body: JSON.stringify({ path: form.path, who: event.who || '', what: event.what || '' }) }))
    notice.value = `事件 ${eventSeq} 已保存`
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

async function saveChapter(eventSeq, chapterSeq) {
  const unit = currentUnit.value
  if (!ensurePath() || !unit) return
  const event = (unit.segments || []).find(item => item.seq === eventSeq)
  const chapter = (event?.chapters || []).find(item => item.seq === chapterSeq)
  if (!chapter) return
  clearMessage(); busy.value = '正在保存章节'
  try {
    applyBook(await api(`/api/units/${unit.id}/segments/${eventSeq}/chapters/${chapterSeq}`, { method: 'PUT', body: JSON.stringify({ path: form.path, who: chapter.who || '', what: chapter.what || '' }) }))
    notice.value = `第 ${chapterSeq} 章已保存`
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

async function addChapter(eventSeq) {
  const unit = currentUnit.value
  if (!ensurePath() || !unit) return
  clearMessage(); busy.value = '正在新增章节'
  try {
    applyBook(await api(`/api/units/${unit.id}/segments/${eventSeq}/chapters`, { method: 'POST', body: JSON.stringify({ path: form.path, who: '', what: '' }) }))
    notice.value = '已新增一章，请填写谁做什么并保存'
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

async function deleteChapter(eventSeq, chapterSeq) {
  const unit = currentUnit.value
  if (!ensurePath() || !unit) return
  if (!window.confirm(`确定删除第 ${chapterSeq} 章吗？此操作不可恢复。`)) return
  clearMessage(); busy.value = '正在删除章节'
  try {
    applyBook(await api(`/api/units/${unit.id}/segments/${eventSeq}/chapters/${chapterSeq}?path=${encodeURIComponent(form.path)}`, { method: 'DELETE' }))
    notice.value = `第 ${chapterSeq} 章已删除`
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

function toggleCarryUnit(id, checked) {
  const unit = currentUnit.value
  if (!unit) return
  const list = unit.carry_units || []
  if (checked && !list.includes(id)) unit.carry_units = [...list, id]
  else if (!checked) unit.carry_units = list.filter(item => item !== id)
  saveCarryUnits()
}
async function saveCarryUnits() {
  const unit = currentUnit.value
  if (!unit) return
  try {
    await api(`/api/units/${unit.id}/working`, {
      method: 'PUT',
      body: JSON.stringify({ path: form.path, carry_units: unit.carry_units || [] }),
    })
  } catch (e) { error.value = e.message }
}

const carryFacts = computed(() => {
  const facts = book.graph?.facts || []
  const currentSeq = Number(currentUnit.value?.seq || 0)
  return facts
    .filter(fact => {
      const seq = Number(fact.unit_seq)
      return Number.isFinite(seq) && seq > 0 && seq < currentSeq
    })
    .sort((a, b) => (Number(a.unit_seq) || 0) - (Number(b.unit_seq) || 0))
})
function toggleCarryFact(id, checked) {
  const unit = currentUnit.value
  if (!unit) return
  const list = unit.carry_facts || []
  if (checked && !list.includes(id)) unit.carry_facts = [...list, id]
  else if (!checked) unit.carry_facts = list.filter(item => item !== id)
  saveCarryFacts()
}
async function saveCarryFacts() {
  const unit = currentUnit.value
  if (!unit) return
  try {
    await api(`/api/units/${unit.id}/working`, {
      method: 'PUT',
      body: JSON.stringify({ path: form.path, carry_facts: unit.carry_facts || [] }),
    })
  } catch (e) { error.value = e.message }
}

async function generateDirectory() {
  const unit = currentUnit.value
  if (!unit || !ensurePath()) return
  clearMessage(); busy.value = '正在准备请求预览…'
  try {
    await persistUnit()
    const payload = { path: form.path, guidance: form.guidance, history: unitHistory() }
    const previewData = await requestPromptPreview(`/api/units/${unit.id}/directory-options`, payload)
    busy.value = ''
    openPromptPreview({
      title: '生成大事件目录',
      kind: previewData.kind || 'directory',
      label: previewData.label || '大事件目录：A/B/C 方案',
      request: previewData.request || '',
      prompt: previewData.prompt || '',
      onSend: async () => {
        await runTaskStrict(
          () => api(`/api/units/${unit.id}/directory-options`, { method: 'POST', body: JSON.stringify(payload) }),
          '正在生成大事件目录', '大事件目录已生成，选一套采用'
        )
      },
    })
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
function choosePlan(key) {
  const unit = currentUnit.value
  if (!unit) return
  return runTask(
    () => api(`/api/units/${unit.id}/directory`, { method: 'PUT', body: JSON.stringify({ path: form.path, plan_key: key }) }),
    '正在采用大事件目录', `已采用大事件目录 ${key}`
  )
}
async function generateEventChapters(eventSeq) {
  const unit = currentUnit.value
  if (!unit || !ensurePath()) return
  clearMessage(); busy.value = '正在准备请求预览…'
  try {
    await persistUnit()
    const payload = { path: form.path, history: unitHistory() }
    const previewData = await requestPromptPreview(`/api/units/${unit.id}/segments/${eventSeq}/chapter-options`, payload)
    busy.value = ''
    openPromptPreview({
      title: `拆解事件 ${eventSeq} 的章节`,
      kind: previewData.kind || 'chapter-options',
      label: previewData.label || `事件${eventSeq}：拆章节方案`,
      request: previewData.request || '',
      prompt: previewData.prompt || '',
      onSend: async () => {
        await runTaskStrict(
          () => api(`/api/units/${unit.id}/segments/${eventSeq}/chapter-options`, { method: 'POST', body: JSON.stringify(payload) }),
          `正在拆解事件 ${eventSeq}`, `事件 ${eventSeq} 的章节方案已生成`
        )
      },
    })
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
function chooseEventChapterPlan(eventSeq, key) {
  const unit = currentUnit.value
  if (!unit) return
  return runTask(
    () => api(`/api/units/${unit.id}/segments/${eventSeq}/chapters`, { method: 'PUT', body: JSON.stringify({ path: form.path, plan_key: key }) }),
    '正在采用章节拆分方案', `事件 ${eventSeq} 已采用方案 ${key}`
  )
}

const merging = ref(false)

async function mergeRoute() {
  const unit = currentUnit.value
  if (!unit || !ensurePath() || merging.value) return
  merging.value = true
  try {
    await runTask(
      () => api(`/api/units/${unit.id}/merge-route`, { method: 'POST', body: JSON.stringify({ path: form.path, guidance: form.guidance, history: unitHistory() }) }),
      '正在校验并合并整条路线…', '完整路线已校验合并'
    )
    nextTick(() => {
      const target = document.querySelector('.merged-route')
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    })
  } finally {
    merging.value = false
  }
}

const batchJob = reactive({ active: false, done: 0, total: 0, detail: '', message: '', autoSelect: false })
let batchPollTimer = null

function stopBatchPoll() {
  if (batchPollTimer) { clearInterval(batchPollTimer); batchPollTimer = null }
  batchJob.active = false
}

async function pollBatchJob() {
  const unit = currentUnit.value
  if (!unit) { stopBatchPoll(); return }
  try {
    const data = await api(`/api/units/${unit.id}/auto-outline?path=${encodeURIComponent(form.path)}`)
    const job = data.job || {}
    batchJob.done = Number(job.done) || 0
    batchJob.total = Number(job.total) || 0
    batchJob.detail = job.detail || ''
    batchJob.message = job.message || ''
    batchJob.active = ['queued', 'running'].includes(job.status)
    if (job.status === 'error') {
      stopBatchPoll()
      error.value = job.message || job.detail || '批量拆分失败'
      finishTraceIfIdle()
    } else if (job.status === 'done') {
      stopBatchPoll()
      await applyBook(await api(`/api/book?path=${encodeURIComponent(form.path)}`))
      notice.value = job.detail || '所有事件的章节方案已生成'
      finishTraceIfIdle()
      if (batchJob.autoSelect) {
        notice.value = '已按评分自动选定整条路线，接下来自动校验合并并给章节起名…'
        await mergeRoute()
        batchJob.autoSelect = false
      }
    }
  } catch (e) {
    stopBatchPoll()
    error.value = e.message
  }
}

async function batchGenerateChapters(force = false, autoSelect = false) {
  const unit = currentUnit.value
  if (!unit || !ensurePath()) return
  clearMessage()
  openTrace()
  try {
    const data = await api(`/api/units/${unit.id}/batch-outline`, { method: 'POST', body: JSON.stringify({ path: form.path, guidance: form.guidance, history: unitHistory(), workers: 4, force, auto_select: autoSelect }) })
    stopBatchPoll()
    batchJob.active = true
    batchJob.autoSelect = !!autoSelect
    batchJob.done = Number((data.job || {}).done) || 0
    batchJob.total = Number((data.job || {}).total) || 0
    batchJob.detail = (data.job || {}).detail || ''
    batchJob.message = ''
    notice.value = autoSelect ? '已开始：并行拆分章节并给每套方案评分，完成后自动选定最高分路线' : '已开始并行拆分：所有事件同时生成章节方案'
    batchPollTimer = setInterval(pollBatchJob, 2500)
  } catch (e) { error.value = e.message }
}

function planWordRatioSum(plan) {
  let sum = 0
  for (const step of (plan?.steps || [])) sum += Number(step.word_ratio) || 0
  return sum
}

function chapterOptionsReady() {
  const unit = currentUnit.value
  return !!unit?.chosen_plan && (unit?.segments || []).some(event => (event.chapter_options || []).length)
}

const routeGrid = computed(() => {
  const unit = currentUnit.value
  if (!unit) return { keys: [], rows: [] }
  const keys = []
  const seen = new Set()
  for (const event of (unit.segments || [])) {
    for (const plan of (event.chapter_options || [])) {
      const key = String(plan.key || '')
      if (key && !seen.has(key)) { seen.add(key); keys.push(key) }
    }
  }
  return { keys, rows: unit.segments || [] }
})

function routePlanOf(event, key) {
  return (event.chapter_options || []).find(plan => String(plan.key) === String(key)) || null
}

function routeLinked(upper, lower, key) {
  return !!upper?.chosen_chapter_plan && String(upper.chosen_chapter_plan) === String(key)
    && !!lower?.chosen_chapter_plan && String(lower.chosen_chapter_plan) === String(key)
}
async function generateDesign(eventSeq, chapterSeq) {
  const unit = currentUnit.value
  if (!unit || !ensurePath()) return
  clearMessage(); busy.value = '正在准备请求预览…'
  try {
    await persistUnit()
    const payload = { path: form.path, history: unitHistory() }
    const previewData = await requestPromptPreview(`/api/units/${unit.id}/segments/${eventSeq}/chapters/${chapterSeq}/design-options`, payload)
    busy.value = ''
    openPromptPreview({
      title: `设计第 ${chapterSeq} 章内容`,
      kind: previewData.kind || 'design',
      label: previewData.label || `事件${eventSeq} 第${chapterSeq}章：内容设计方案`,
      request: previewData.request || '',
      prompt: previewData.prompt || '',
      onSend: async () => {
        await runTaskStrict(
          () => api(`/api/units/${unit.id}/segments/${eventSeq}/chapters/${chapterSeq}/design-options`, { method: 'POST', body: JSON.stringify(payload) }),
          `正在设计第 ${chapterSeq} 章`, `第 ${chapterSeq} 章的内容方案已生成`
        )
      },
    })
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
function chooseDesign(eventSeq, chapterSeq, key) {
  const unit = currentUnit.value
  if (!unit) return
  return runTask(
    () => api(`/api/units/${unit.id}/segments/${eventSeq}/chapters/${chapterSeq}/design`, { method: 'PUT', body: JSON.stringify({ path: form.path, option_key: key }) }),
    '正在确定内容方案', `第 ${chapterSeq} 章已采用方案 ${key}`
  )
}
const draftInstructionKey = (eventSeq, chapterSeq) => `${eventSeq}:${chapterSeq}`
async function generateDraft(eventSeq, chapterSeq) {
  const unit = currentUnit.value
  if (!unit || !ensurePath()) return
  clearMessage(); busy.value = '正在准备请求预览…'
  try {
    await persistUnit()
    const payload = { path: form.path, history: unitHistory(), extra_instruction: draftInstructions[draftInstructionKey(eventSeq, chapterSeq)] || '' }
    const previewData = await requestPromptPreview(`/api/units/${unit.id}/segments/${eventSeq}/chapters/${chapterSeq}/draft`, payload)
    busy.value = ''
    openPromptPreview({
      title: `写第 ${chapterSeq} 章正文`,
      kind: previewData.kind || 'draft',
      label: previewData.label || `事件${eventSeq} 第${chapterSeq}章：写正文`,
      request: previewData.request || '',
      prompt: previewData.prompt || '',
      onSend: async () => {
        await runTaskStrict(
          () => api(`/api/units/${unit.id}/segments/${eventSeq}/chapters/${chapterSeq}/draft`, { method: 'POST', body: JSON.stringify(payload) }),
          `正在写第 ${chapterSeq} 章正文`, `第 ${chapterSeq} 章正文已生成`
        )
      },
    })
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function saveDraft(eventSeq, chapterSeq, text) {
  const unit = currentUnit.value
  if (!unit) return
  try {
    applyBook(await api(`/api/units/${unit.id}/segments/${eventSeq}/chapters/${chapterSeq}/draft`, { method: 'PUT', body: JSON.stringify({ path: form.path, text }) }))
    notice.value = '正文已保存'
  } catch (e) { error.value = e.message }
}
async function saveSummary() {
  const unit = currentUnit.value
  if (!unit) return
  try {
    applyBook(await api(`/api/units/${unit.id}/summary`, { method: 'PUT', body: JSON.stringify({ path: form.path, text: unit.unit_summary }) }))
    notice.value = '章节摘要已保存'
  } catch (e) { error.value = e.message }
}

function ensureSnapshot(unit) {
  if (!unit) return
  unit.snapshot = unit.snapshot || { beats: [], character_states: [], hooks: [] }
  unit.snapshot.beats = unit.snapshot.beats || []
  unit.snapshot.character_states = unit.snapshot.character_states || []
  unit.snapshot.hooks = unit.snapshot.hooks || []
}
function addSnapshotField(key) {
  const unit = currentUnit.value
  if (!unit) return
  ensureSnapshot(unit)
  unit.snapshot[key].push('')
}
function removeSnapshotField(key, index) {
  const unit = currentUnit.value
  if (!unit) return
  ensureSnapshot(unit)
  unit.snapshot[key].splice(index, 1)
}
async function saveSnapshot() {
  const unit = currentUnit.value
  if (!unit) return
  ensureSnapshot(unit)
  try {
    applyBook(await api(`/api/units/${unit.id}/snapshot`, { method: 'PUT', body: JSON.stringify({ path: form.path, snapshot: unit.snapshot }) }))
    notice.value = '前情快照已保存'
  } catch (e) { error.value = e.message }
}
async function generateSnapshot() {
  const unit = currentUnit.value
  if (!unit || !ensurePath()) return
  clearMessage(); busy.value = '正在准备快照草稿预览…'
  try {
    const payload = { path: form.path }
    const previewData = await requestPromptPreview(`/api/units/${unit.id}/snapshot`, payload)
    busy.value = ''
    openPromptPreview({
      title: '生成章节快照草稿',
      kind: previewData.kind || 'snapshot',
      label: previewData.label || '章节收尾：前情快照',
      request: previewData.request || '',
      prompt: previewData.prompt || '',
      onSend: async () => {
        busy.value = '正在生成快照草稿'
        try {
          const data = await api(`/api/units/${unit.id}/snapshot`, { method: 'POST', body: JSON.stringify(payload) })
          ensureSnapshot(unit)
          Object.assign(unit.snapshot, data.snapshot || {})
          notice.value = '快照草稿已生成，请检查后再保存'
        } catch (e) { error.value = e.message; throw e }
        finally { busy.value = '' }
      },
    })
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

function ensureGraph() {
  book.graph = book.graph || { nodes: [], edges: [], events: [], facts: [] }
  book.graph.nodes = book.graph.nodes || []
  book.graph.edges = book.graph.edges || []
  book.graph.events = book.graph.events || []
  book.graph.facts = book.graph.facts || []
}
function graphNodeName(id) {
  ensureGraph()
  return book.graph.nodes.find(node => node.id === id)?.name || id || ''
}
const graphKindColors = {
  人物: '#c4572c',
  地点: '#2f7d55',
  组织: '#3b5f8e',
  物品: '#7a4f9c',
  悬念: '#b03a34',
}
function graphKindColor(kind) {
  return graphKindColors[kind] || '#6f675c'
}
function truncateGraphLabel(text) {
  const value = (text || '未命名').trim() || '未命名'
  return value.length > 8 ? value.slice(0, 8) + '…' : value
}
const graphView = computed(() => {
  ensureGraph()
  const nodes = (book.graph.nodes || []).map(node => ({ ...node, name: (node.name || '').trim() || '未命名' }))
  const byId = {}
  nodes.forEach(node => { byId[node.id] = node })

  const cx = 460
  const cy = 250
  const rx = 340
  const ry = 190
  const placed = nodes.map((node, index) => {
    const angle = -Math.PI / 2 + (nodes.length > 1 ? (index * Math.PI * 2) / nodes.length : 0)
    return { ...node, x: cx + Math.cos(angle) * rx, y: cy + Math.sin(angle) * ry }
  })
  const posById = {}
  placed.forEach(node => { posById[node.id] = node })

  const pairIndex = {}
  const edges = (book.graph.edges || [])
    .filter(edge => edge.source && edge.target && byId[edge.source] && byId[edge.target])
    .map(edge => {
      const type = (edge.type || '').trim() || '关系'
      const source = posById[edge.source]
      const target = posById[edge.target]
      let path = ''
      let labelX = 0
      let labelY = 0
      if (source.id === target.id) {
        path = `M ${source.x} ${source.y - 20} C ${source.x - 54} ${source.y - 82}, ${source.x + 54} ${source.y - 82}, ${source.x} ${source.y - 20}`
        labelX = source.x
        labelY = source.y - 82
      } else {
        const dx = target.x - source.x
        const dy = target.y - source.y
        const len = Math.hypot(dx, dy)
        if (len < 1) return null
        const ux = dx / len
        const uy = dy / len
        const px = -uy
        const py = ux
        const key = [source.id, target.id].sort().join('__')
        const order = pairIndex[key] || 0
        pairIndex[key] = order + 1
        const bend = (order % 2 === 0 ? 1 : -1) * (54 + Math.floor(order / 2) * 28)
        const mx = (source.x + target.x) / 2 + px * bend
        const my = (source.y + target.y) / 2 + py * bend
        const ex = target.x - ux * 24
        const ey = target.y - uy * 24
        path = `M ${source.x} ${source.y} Q ${mx} ${my} ${ex} ${ey}`
        labelX = source.x * 0.25 + mx * 0.5 + target.x * 0.25
        labelY = source.y * 0.25 + my * 0.5 + target.y * 0.25
      }
      return { ...edge, type, path, labelX, labelY, labelW: Math.max(28, type.length * 13 + 14) }
    })
    .filter(Boolean)

  return { nodes: placed, edges }
})
function addGraphNode() {
  ensureGraph()
  book.graph.nodes.push({ id: `n${Date.now().toString(36)}`, kind: '人物', name: '', note: '', status: '活跃' })
}
function removeGraphNode(index) {
  ensureGraph()
  const node = book.graph.nodes[index]
  book.graph.nodes.splice(index, 1)
  if (node?.id) book.graph.edges = book.graph.edges.filter(edge => edge.source !== node.id && edge.target !== node.id)
}
function addGraphEdge() {
  ensureGraph()
  book.graph.edges.push({ id: `e${Date.now().toString(36)}`, source: '', target: '', type: '', note: '', status: '' })
}
function removeGraphEdge(index) {
  ensureGraph()
  book.graph.edges.splice(index, 1)
}
function addGraphEvent() {
  ensureGraph()
  book.graph.events.push({ id: `t${Date.now().toString(36)}`, unit_seq: currentUnit.value?.seq || book.units.length || '', title: '', summary: '' })
}
function removeGraphEvent(index) {
  ensureGraph()
  book.graph.events.splice(index, 1)
}
function addGraphFact() {
  ensureGraph()
  book.graph.facts.push({ id: `f${Date.now().toString(36)}`, unit_seq: currentUnit.value?.seq || book.units.length || '', text: '', entities: [] })
}
function removeGraphFact(index) {
  ensureGraph()
  book.graph.facts.splice(index, 1)
}
function factEntityText(fact) {
  ensureGraph()
  return (fact.entities || []).join('、')
}
function setFactEntities(fact, value) {
  ensureGraph()
  fact.entities = String(value || '')
    .split(/[、,，\s]+/)
    .map(item => item.trim())
    .filter(Boolean)
}
async function saveGraph() {
  if (!ensurePath()) return
  ensureGraph()
  try {
    applyBook(await api('/api/graph', { method: 'PUT', body: JSON.stringify({ path: form.path, graph: book.graph }) }))
    notice.value = '知识图谱已保存'
  } catch (e) { error.value = e.message }
}
async function generateGraph() {
  if (!ensurePath()) return
  clearMessage(); busy.value = '正在准备图谱草稿预览…'
  try {
    const payload = { path: form.path }
    const previewData = await requestPromptPreview('/api/graph/draft', payload)
    busy.value = ''
    openPromptPreview({
      title: '起草知识图谱',
      kind: previewData.kind || 'graph',
      label: previewData.label || '知识图谱：起草更新',
      request: previewData.request || '',
      prompt: previewData.prompt || '',
      onSend: async () => {
        busy.value = '正在生成图谱草稿'
        try {
          const data = await api('/api/graph/draft', { method: 'POST', body: JSON.stringify(payload) })
          book.graph = data.graph || { nodes: [], edges: [], events: [] }
          notice.value = '图谱草稿已载入，请检查后再保存'
        } catch (e) { error.value = e.message; throw e }
        finally { busy.value = '' }
      },
    })
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

async function atomizeUnit() {
  const unit = currentUnit.value
  if (!unit || !ensurePath()) return
  clearMessage(); busy.value = '正在准备拆分预览…'
  try {
    const payload = { path: form.path }
    const previewData = await requestPromptPreview(`/api/units/${unit.id}/atomize`, payload)
    busy.value = ''
    openPromptPreview({
      title: 'GraphRAG 拆分小事件',
      kind: previewData.kind || 'atomize',
      label: previewData.label || 'GraphRAG：拆分小事件',
      request: previewData.request || '',
      prompt: previewData.prompt || '',
      onSend: async () => {
        busy.value = '正在拆分小事件'
        try {
          const data = await api(`/api/units/${unit.id}/atomize`, { method: 'POST', body: JSON.stringify(payload) })
          ensureGraph()
          book.graph.facts = book.graph.facts.filter(fact => String(fact.unit_seq) !== String(data.unit_seq))
          for (const fact of (data.facts || [])) book.graph.facts.push(fact)
          notice.value = `已拆出 ${(data.facts || []).length} 条小事件，请检查后再保存图谱`
        } catch (e) { error.value = e.message; throw e }
        finally { busy.value = '' }
      },
    })
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}

function unitStatus(unit) {
  if (!unit) return '待初始化'
  if (eventChapters(unit).some(({ chapter }) => (chapter.draft || '').trim())) return '有正文'
  if (unit.chosen_plan) return '已定大事件目录'
  return '待创作'
}
function worldReady() {
  return !![book.world.premise, book.world.era_place, book.world.rules, book.world.tone].some(value => (value || '').trim())
}
function settingsReady() {
  return bookScaleReady() && worldReady()
}
function eventChapters(unit) {
  const list = []
  for (const event of (unit?.segments || [])) {
    for (const chapter of (event.chapters || [])) {
      list.push({ event, chapter })
    }
  }
  return list
}
function unitDraftReady(unit) {
  return eventChapters(unit).some(({ chapter }) => (chapter.draft || '').trim())
}
function routeChosen(unit) {
  const events = (unit?.segments || [])
  return events.length > 0 && events.every(event => !!event?.chosen_chapter_plan)
}
function unitProgress(unit) {
  if (!unit) return 0
  const checks = [
    settingsReady(),
    !!unit?.chosen_plan,
    (unit?.segments || []).some(event => (event.chapter_options || []).length || (event.chapters || []).length),
    routeChosen(unit),
    !!((unit.merged_route || {}).chapters || []).length,
  ]
  return Math.round((checks.filter(Boolean).length / checks.length) * 100)
}
const currentFlow = computed(() => {
  const unit = currentUnit.value
  if (!unit) return []
  const steps = [
    { key: 'plan', label: '大事件目录', hint: '与设定一次生成，采用即选定', done: !!unit.chosen_plan },
    { key: 'options', label: '章节方案', hint: '一键并行拆分全部事件', done: (unit.segments || []).some(event => (event.chapter_options || []).length || (event.chapters || []).length) },
    { key: 'route', label: '路线选择', hint: '逐事件选定章节路线', done: routeChosen(unit) },
    { key: 'merge', label: '校验合并', hint: 'AI 查重补缺，给出合并方案', done: !!((unit.merged_route || {}).chapters || []).length },
  ]
  const firstPending = steps.findIndex(step => !step.done)
  return steps.map((step, index) => ({
    ...step,
    state: step.done ? 'done' : (index === firstPending ? 'active' : 'todo'),
  }))
})
function draftWordCount() {
  const unit = currentUnit.value
  if (!unit) return 0
  return eventChapters(unit).reduce((total, { chapter }) => total + String(chapter?.draft || '').replace(/\s/g, '').length, 0)
}
function fullDraftText() {
  const unit = currentUnit.value
  if (!unit) return ''
  let global = 0
  const blocks = []
  for (const event of (unit.segments || [])) {
    for (const chapter of (event.chapters || [])) {
      global += 1
      const head = `\n\n\n第 ${global} 章 · ${chapter.who || '（未写人物）'} → ${chapter.what || '（未写动作）'}\n`
      blocks.push(head + (chapter.draft || '').trim())
    }
  }
  return blocks.join('').trim()
}
async function copyFullText() {
  const text = fullDraftText()
  if (!text) { notice.value = '当前章节还没有正文'; return }
  try {
    await navigator.clipboard.writeText(text)
    notice.value = '全文已复制到剪贴板'
  } catch {
    window.prompt('复制失败，请手动复制下面的正文', text)
  }
}
function downloadFullText() {
  const unit = currentUnit.value
  const text = fullDraftText()
  if (!text) { notice.value = '全书还没有正文'; return }
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${book.title || unit.title || '全书'}.txt`
  link.click()
  URL.revokeObjectURL(url)
}

// 设置面板
function choosePreset(name) {
  selectedPreset.value = name
  Object.assign(draftPreset, config.value.llm_configs?.[name] || {})
}
function addPreset() {
  let base = '新模型配置', index = 1, name = base
  while (config.value.llm_configs?.[name]) name = `${base} ${++index}`
  config.value.llm_configs ||= {}
  config.value.llm_configs[name] = { api_key: '', base_url: '', interface_format: 'OpenAI', model_name: '', temperature: 0.7, max_tokens: 8192, timeout: 600 }
  choosePreset(name)
}
async function saveSettings() {
  if (!selectedPreset.value) return
  config.value.llm_configs ||= {}
  config.value.llm_configs[selectedPreset.value] = { ...config.value.llm_configs[selectedPreset.value], ...draftPreset }
  config.value.last_llm_config_name = selectedPreset.value
  config.value.choose_configs ||= {}
  try {
    await api('/api/config', { method: 'PUT', body: JSON.stringify(config.value) })
    showSettings.value = false; notice.value = '模型配置已保存'
  } catch (e) { error.value = e.message }
}

async function loadConfig() {
  config.value = await api('/api/config')
  config.value.llm_configs ||= {}
  config.value.choose_configs ||= {}
  const fallback = config.value.last_llm_config_name || Object.keys(config.value.llm_configs)[0] || ''
  if (!config.value.choose_configs.chapter_outline_llm) config.value.choose_configs.chapter_outline_llm = fallback
  if (!config.value.choose_configs.prompt_draft_llm) config.value.choose_configs.prompt_draft_llm = fallback
  const storedPath = String(config.value.other_params?.filepath || '').trim()
  form.path = /^[A-Za-z]:[\\/]/.test(storedPath) ? '' : storedPath
  if (storedPath && !form.path) notice.value = '检测到旧版 Windows 项目路径，当前设备不适用，请重新填写项目路径'
  choosePreset(fallback)
  if (form.path) await loadProject()
}

// 选中文字 → 自己问
function openAsk() {
  if (!sel.text) return
  ask.action = 'answer'
  ask.text = sel.text
  ask.scope = sel.scope || sel.text
  ask.question = ''
  ask.input = ''
  ask.result = ''
  ask.messages = []
  ask.show = true
  sel.visible = false
}
function closeAsk() { ask.show = false }
const CHAT_LOG_KEY = 'ainovel_chat_history_v1'
function restoreChatLogs() {
  try {
    const store = JSON.parse(localStorage.getItem(CHAT_LOG_KEY) || '{}')
    chatLogs.value = (form.path && store?.[form.path]) || {}
  } catch { chatLogs.value = {} }
}
function persistChatLogs() {
  try {
    const store = JSON.parse(localStorage.getItem(CHAT_LOG_KEY) || '{}')
    store[form.path] = chatLogs.value
    localStorage.setItem(CHAT_LOG_KEY, JSON.stringify(store))
  } catch { /* localStorage 不可用时静默 */ }
}
function rememberChat(message) {
  const key = currentUnit.value?.id || '__book__'
  const list = (chatLogs.value[key] || []).filter(item => item && typeof item.content === 'string')
  chatLogs.value = { ...chatLogs.value, [key]: [...list, message].slice(-40) }
  persistChatLogs()
}
function unitHistory() {
  const unitKey = currentUnit.value?.id || '__book__'
  const unitMsgs = chatLogs.value[unitKey] || []
  const bookMsgs = chatLogs.value['__book__'] || []
  return [...bookMsgs, ...unitMsgs].slice(-16)
}
async function submitAsk() {
  if (!ask.text.trim()) { error.value = '没有选中的文字'; return }
  if (!form.path.trim()) { error.value = '请先填写项目保存路径'; return }
  const content = (ask.messages.length ? ask.input : ask.question).trim()
  if (!content) return
  const nextMessages = [...ask.messages, { role: 'user', content }]
  const payload = {
    path: form.path, selected: ask.text, action: ask.action, messages: nextMessages,
    unit_id: currentUnit.value?.id || '',
    scope_text: ask.scope || ask.text,
  }
  clearMessage()
  busy.value = '正在准备请求预览…'
  try {
    const previewData = await requestPromptPreview('/api/ask', payload)
    busy.value = ''
    openPromptPreview({
      title: '问 AI',
      kind: previewData.kind || 'ask',
      label: previewData.label || '选中文字：多轮问 AI',
      request: previewData.request || '',
      prompt: previewData.prompt || '',
      onSend: async () => {
        ask.busy = true
        try {
          const data = await api('/api/ask', { method: 'POST', body: JSON.stringify(payload) })
          ask.result = data.result || ''
          ask.usedRewrite = /——新版开始——/.test(ask.result)
          const rewriteCall = (data.tool_calls || []).find(call => call.name === 'rewrite_text')
          if (rewriteCall) {
            const args = rewriteCall.arguments || {}
            ask.pendingRewrite = { reason: args.reason || '', replacement: args.replacement || '' }
            ask.pendingRewriteError = ''
            ask.showRewriteConfirm = true
          }
          ask.messages.push({ role: 'user', content })
          rememberChat({ role: 'user', content })
          ask.messages.push({ role: 'assistant', content: ask.result })
          rememberChat({ role: 'assistant', content: ask.result })
          ask.input = ''
        } catch (e) { error.value = e.message; throw e }
        finally { ask.busy = false }
      },
    })
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
function clearAsk() {
  ask.messages = []
  ask.result = ''
  ask.input = ''
  ask.question = ''
  ask.usedRewrite = false
  ask.pendingRewrite = null
  ask.showRewriteConfirm = false
}
async function copyAskResult() {
  if (!ask.result) return
  try { await navigator.clipboard.writeText(ask.result); notice.value = '结果已复制' }
  catch { window.prompt('复制失败，请手动复制', ask.result) }
}
function extractRewrite(text) {
  const match = String(text || '').match(/——新版开始——([\s\S]*?)——新版结束——/)
  return match ? match[1].trim() : String(text || '').trim()
}
function targetForElement(el) {
  const holder = el?.closest?.('[data-kind]')
  if (!holder) return null
  const kind = holder.dataset.kind
  if (kind === 'plan-step') return { kind, planKey: holder.dataset.planKey, stepSeq: holder.dataset.stepSeq }
  if (kind === 'plan-thesis') return { kind, planKey: holder.dataset.planKey }
  if (kind === 'chapter-plan-step') return { kind, eventSeq: holder.dataset.eventSeq, planKey: holder.dataset.planKey, stepSeq: holder.dataset.stepSeq }
  if (kind === 'chapter-plan-thesis') return { kind, eventSeq: holder.dataset.eventSeq, planKey: holder.dataset.planKey }
  if (kind === 'event-head') return { kind, eventSeq: holder.dataset.eventSeq }
  if (kind === 'chapter-head') return { kind, eventSeq: holder.dataset.eventSeq, chapterSeq: holder.dataset.chapterSeq }
  if (kind === 'design-field') return { kind, eventSeq: holder.dataset.eventSeq, chapterSeq: holder.dataset.chapterSeq, optionKey: holder.dataset.optionKey, field: holder.dataset.field }
  return null
}
function parseWhoWhat(text, fallbackWho) {
  const cleaned = String(text || '').trim()
  const match = cleaned.match(/^(.*?)\s*(?:→|->)\s*(.*)$/s)
  if (!match) return { who: fallbackWho || '', what: cleaned }
  return { who: (match[1] || fallbackWho || '').trim(), what: (match[2] || '').trim() }
}
function spliceText(full, start, end, replacement) {
  const text = String(full || '')
  const from = Math.max(0, Math.min(Number(start) || 0, text.length))
  const to = Math.max(from, Math.min(Number(end) || from, text.length))
  return text.slice(0, from) + String(replacement || '') + text.slice(to)
}
function applyTargetRewrite(target, replacement) {
  const unit = currentUnit.value
  if (!unit || !target || !replacement) return false
  if (target.kind === 'plan-step') {
    const plan = (unit.directory_options || []).find(item => String(item.key) === String(target.planKey))
    const step = plan?.steps?.find(item => String(item.seq) === String(target.stepSeq))
    if (!step) return false
    const parsed = parseWhoWhat(replacement, step.who)
    step.who = parsed.who
    step.what = parsed.what
    return true
  }
  if (target.kind === 'plan-thesis') {
    const plan = (unit.directory_options || []).find(item => String(item.key) === String(target.planKey))
    if (!plan) return false
    plan.thesis = String(replacement).trim()
    return true
  }
  if (target.kind === 'chapter-plan-step') {
    const event = (unit.segments || []).find(item => String(item.seq) === String(target.eventSeq))
    const plan = (event?.chapter_options || []).find(item => String(item.key) === String(target.planKey))
    const step = plan?.steps?.find(item => String(item.seq) === String(target.stepSeq))
    if (!step) return false
    const parsed = parseWhoWhat(replacement, step.who)
    step.who = parsed.who
    step.what = parsed.what
    return true
  }
  if (target.kind === 'chapter-plan-thesis') {
    const event = (unit.segments || []).find(item => String(item.seq) === String(target.eventSeq))
    const plan = (event?.chapter_options || []).find(item => String(item.key) === String(target.planKey))
    if (!plan) return false
    plan.thesis = String(replacement).trim()
    return true
  }
  if (target.kind === 'design-field') {
    const event = (unit.segments || []).find(item => String(item.seq) === String(target.eventSeq))
    const chapter = event?.chapters?.find(item => String(item.seq) === String(target.chapterSeq))
    const option = chapter?.design_options?.find(item => String(item.key) === String(target.optionKey))
    if (!option) return false
    if (target.field === 'title') {
      option.title = String(replacement).replace(/^方案\s*\d+\s*·\s*/, '').trim()
    } else if (target.field === 'summary' || target.field === 'content') {
      option[target.field] = String(replacement).trim()
    } else {
      return false
    }
    return true
  }
  if (target.kind === 'event-head') {
    const event = (unit.segments || []).find(item => String(item.seq) === String(target.eventSeq))
    if (!event) return false
    const parsed = parseWhoWhat(replacement, event.who)
    event.who = parsed.who
    event.what = parsed.what
    return true
  }
  if (target.kind === 'chapter-head') {
    const event = (unit.segments || []).find(item => String(item.seq) === String(target.eventSeq))
    const chapter = event?.chapters?.find(item => String(item.seq) === String(target.chapterSeq))
    if (!chapter) return false
    const parsed = parseWhoWhat(replacement, chapter.who)
    chapter.who = parsed.who
    chapter.what = parsed.what
    return true
  }
  if (target.kind === 'segment-head') {
    const segment = (unit.segments || []).find(item => String(item.seq) === String(target.segSeq))
    if (!segment) return false
    const parsed = parseWhoWhat(replacement, segment.who)
    segment.who = parsed.who
    segment.what = parsed.what
    return true
  }
  return false
}
async function persistCurrentUnitWorking() {
  const unit = currentUnit.value
  if (!unit) return
  applyBook(await api(`/api/units/${unit.id}/working`, {
    method: 'PUT',
    body: JSON.stringify({
      path: form.path,
      directory_options: unit.directory_options || [],
      chosen_plan: unit.chosen_plan || '',
      segments: unit.segments || [],
      unit_summary: unit.unit_summary || '',
    }),
  }))
}
function targetStepIdentity(target) {
  const unit = currentUnit.value
  if (!unit) return ''
  if (target.kind === 'plan-step') {
    const plan = (unit.directory_options || []).find(item => String(item.key) === String(target.planKey))
    const step = plan?.steps?.find(item => String(item.seq) === String(target.stepSeq))
    return step?.who || ''
  }
  if (target.kind === 'chapter-plan-step') {
    const event = (unit.segments || []).find(item => String(item.seq) === String(target.eventSeq))
    const plan = (event?.chapter_options || []).find(item => String(item.key) === String(target.planKey))
    const step = plan?.steps?.find(item => String(item.seq) === String(target.stepSeq))
    return step?.who || ''
  }
  if (target.kind === 'segment-head') {
    const segment = (unit.segments || []).find(item => String(item.seq) === String(target.segSeq))
    return segment?.who || ''
  }
  if (target.kind === 'event-head') {
    const event = (unit.segments || []).find(item => String(item.seq) === String(target.eventSeq))
    return event?.who || ''
  }
  if (target.kind === 'chapter-head') {
    const event = (unit.segments || []).find(item => String(item.seq) === String(target.eventSeq))
    const chapter = event?.chapters?.find(item => String(item.seq) === String(target.chapterSeq))
    return chapter?.who || ''
  }
  return ''
}
function targetMatchesLine(target, line) {
  const parsed = parseWhoWhat(line, '')
  const lineWho = (parsed.who || '').trim()
  const targetWho = targetStepIdentity(target).trim()
  if (!lineWho || !targetWho) return false
  return targetWho.includes(lineWho) || lineWho.includes(targetWho)
}
async function applyAskResult(replacementOverride) {
  const replacement = (replacementOverride ?? extractRewrite(ask.result)).trim()
  if (!replacement) { error.value = '没有可应用的文本'; return false }
  if (sel.ref && typeof sel.ref.value === 'string') {
    sel.ref.value = spliceText(sel.ref.value, sel.start, sel.end, replacement)
    sel.ref.dispatchEvent(new Event('input', { bubbles: true }))
    sel.ref = null
    sel.target = null
    sel.targets = []
    ask.show = false
    notice.value = '已替换原文，记得保存'
    return true
  }
  if (!sel.targets.length && sel.range) {
    sel.targets = targetsInRange(sel.range)
    if (sel.targets.length === 1) sel.target = sel.targets[0]
  }
  if (sel.targets.length) {
    const targets = [...sel.targets]
    const lines = replacement.split(/\r?\n/).map(line => line.trim()).filter(Boolean)
    if (!lines.length) {
      error.value = '替换文本为空，无法写入'
      return false
    }
    let pairs = []
    if (targets.length === 1) {
      const spliced = (sel.full && sel.end > sel.start)
        ? spliceText(sel.full, sel.start, sel.end, replacement)
        : replacement
      pairs = [{ target: targets[0], replacement: spliced }]
    } else if (lines.length === targets.length) {
      pairs = targets.map((target, index) => ({ target, replacement: lines[index] }))
    } else {
      const used = new Set()
      for (const line of lines) {
        let index = targets.findIndex((target, targetIndex) => !used.has(targetIndex) && targetMatchesLine(target, line))
        if (index < 0) index = targets.findIndex((target, targetIndex) => !used.has(targetIndex))
        if (index >= 0) {
          used.add(index)
          pairs.push({ target: targets[index], replacement: line })
        }
      }
    }
    let applied = 0
    for (const pair of pairs) {
      if (applyTargetRewrite(pair.target, pair.replacement)) applied += 1
    }
    if (!applied) { error.value = '没有找到对应的可编辑位置'; return false }
    try {
      await persistCurrentUnitWorking()
      sel.target = null
      sel.targets = []
      sel.range = null
      ask.show = false
      notice.value = '已写入当前章节'
    } catch (e) { error.value = e.message; return false }
    return true
  }
  if (sel.target) {
    if (!applyTargetRewrite(sel.target, replacement)) { error.value = '没有找到对应的可编辑位置'; return false }
    try {
      await persistCurrentUnitWorking()
      sel.target = null
      sel.targets = []
      sel.range = null
      ask.show = false
      notice.value = '已写入当前章节'
    } catch (e) { error.value = e.message; return false }
    return true
  }
  error.value = '当前选区无法直接写入，请复制后再手动保存'
  return false
}
function cancelRewriteConfirm() {
  ask.showRewriteConfirm = false
  ask.pendingRewrite = null
  ask.pendingRewriteError = ''
}
async function confirmRewriteApply() {
  const replacement = (ask.pendingRewrite?.replacement || '').trim()
  if (!replacement) { cancelRewriteConfirm(); return }
  ask.pendingRewriteError = ''
  const applied = await applyAskResult(replacement)
  if (applied) {
    ask.showRewriteConfirm = false
    ask.pendingRewrite = null
    ask.pendingRewriteError = ''
  } else {
    ask.pendingRewriteError = error.value || '写入失败'
  }
}
function scopeForElement(el) {
  const holder = el?.closest?.('.plan, .design, .segment, .event, .chapter, .cast-chip, .step-card')
  if (!holder) return ''
  const text = (holder.innerText || holder.textContent || '').trim()
  return text.replace(/\n{3,}/g, '\n\n')
}
function targetsInRange(range) {
  if (!range) return []
  const targets = []
  document.querySelectorAll('[data-kind]').forEach(el => {
    let intersects = false
    try { intersects = range.intersectsNode ? range.intersectsNode(el) : false } catch { intersects = false }
    if (intersects) {
      const target = targetForElement(el)
      if (target) targets.push(target)
    }
  })
  return targets
}
function fullTextForSelection(el, fallbackText, targets) {
  if (targets.length !== 1) return fallbackText
  const node = el?.closest?.('[data-kind]') || null
  if (!node) return fallbackText
  return emptyRangeText(node) || fallbackText
}
function emptyRangeText(root) {
  if (!root) return ''
  try {
    const probe = document.createRange()
    probe.selectNodeContents(root)
    return probe.toString()
  } catch { return root.innerText || root.textContent || '' }
}
function offsetsInNode(root, range) {
  if (!root || !range) return null
  try {
    const probe = document.createRange()
    probe.selectNodeContents(root)
    probe.setEnd(range.startContainer, range.startOffset)
    const start = probe.toString().length
    probe.setEnd(range.endContainer, range.endOffset)
    const end = probe.toString().length
    return start <= end ? { start, end } : null
  } catch { return null }
}
function setSel(text, refNode, start, end, x, y, scope, target = null, targets = [], range = null, full = '') {
  sel.text = text
  sel.scope = scope || ''
  sel.full = full || text
  sel.ref = refNode
  sel.target = target || null
  sel.targets = targets || []
  sel.range = range ? markRaw(range) : null
  sel.start = start
  sel.end = end
  sel.x = Math.max(8, Math.min(x, window.innerWidth - 260))
  sel.y = Math.max(8, Math.min(y, window.innerHeight - 140))
  sel.visible = true
}
function captureSelectionNow(event) {
  const active = document.activeElement
  if (active && (active.tagName === 'TEXTAREA' || (active.tagName === 'INPUT' && /text|search|url|password/i.test(active.type || '')))) {
    const start = typeof active.selectionStart === 'number' ? active.selectionStart : 0
    const end = typeof active.selectionEnd === 'number' ? active.selectionEnd : 0
    const text = (active.value || '').slice(start, end).trim()
    if (text) { setSel(text, active, start, end, event.clientX, event.clientY + 16, scopeForElement(active), null, [], null, active.value || ''); return }
  }
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed) { sel.visible = false; return }
  const text = selection.toString().trim()
  if (!text) { sel.visible = false; return }
  const range = selection.getRangeAt(0)
  const rect = range.getBoundingClientRect()
  const ancestor = range.commonAncestorContainer
  const containEl = ancestor?.nodeType === 1 ? ancestor : ancestor?.parentElement
  const targets = targetsInRange(range)
  const full = fullTextForSelection(containEl, text, targets)
  let fullText = full
  let startOffset = 0
  let endOffset = 0
  if (targets.length === 1) {
    const node = containEl?.closest?.('[data-kind]') || null
    const offsets = offsetsInNode(node, range)
    if (offsets && offsets.end > offsets.start) {
      fullText = full
      startOffset = offsets.start
      endOffset = offsets.end
    }
  }
  setSel(text, null, startOffset, endOffset, rect.left + rect.width / 2, rect.bottom + 12, scopeForElement(containEl), targets[0] || null, targets, range.cloneRange(), fullText)
}
function onSelectionMouseUp(event) {
  if (event.target?.closest?.('.ask-toolbar, .modal-backdrop, .trace-panel, .topbar')) return
  requestAnimationFrame(() => captureSelectionNow(event))
}
function onSelectionMouseDown(event) {
  if (!event.target?.closest?.('.ask-toolbar')) sel.visible = false
}

onMounted(() => {
  loadConfig().catch(e => error.value = e.message)
  document.addEventListener('mouseup', onSelectionMouseUp)
  document.addEventListener('mousedown', onSelectionMouseDown)
})
onUnmounted(() => {
  closeTrace()
  stopBatchPoll()
  document.removeEventListener('mouseup', onSelectionMouseUp)
  document.removeEventListener('mousedown', onSelectionMouseDown)
})
</script>

<template>
  <div class="shell">
    <main>
      <header class="topbar">
        <div class="topbar-bottom">
          <div class="page-head">
            <h1>一套流程，从设定到成书</h1>
            <p>上传参考文档、AI 生成设定 → 拆大事件 → 一键章节方案 → 逐事件选路线 → 校验合并。</p>
          </div>
          <div class="project-box">
            <label for="project-path">项目路径</label>
            <div class="project-load">
              <input id="project-path" v-model="form.path" placeholder="例如 ~/我的小说/夜航书店" />
              <button class="primary-button" @click="loadProject">加载项目</button>
            </div>
          </div>
        </div>
      </header>

      <div v-if="!form.path" class="banner hint">还没有项目路径：请先在右上角填一个本地文件夹，之后所有设定和正文都会保存在里面。</div>
      <div v-if="busy" class="banner busy">
        <span class="spinner"></span>
        <span class="banner-body"><strong>{{ busy }}</strong><small>{{ traceStatus }}</small></span>
        <button class="banner-toggle" @click="showTrace ? closeTrace() : openTrace()">{{ showTrace ? '收起日志' : '查看日志' }}</button>
      </div>
      <div v-if="error" class="banner error">{{ error }}</div>
      <div v-if="notice" class="banner success">{{ notice }}</div>

      <!-- ============ 设定 ============ -->
      <section class="setup-page">
        <div class="card onboarding">
          <div class="onboarding-copy">
            <h2>六步开始一本书</h2>
            <p>先上传参考文档（可跳过）并填好全书总字数（必填），让 AI 分三次独立调用生成三套全书方案——每套都含单主线、世界观、人物和大事件目录；经过辩论自动采用一套后，往下拆章节、选路线、校验合并，最后串行写完各章正文。AI 只引用你给的设定，不替你发明。</p>
          </div>
          <ol class="onboarding-steps">
            <li :class="{ done: bookScaleReady() }"><span>1</span><div><strong>上传文档 + 定规模</strong><small>.txt / .md / .epub 可选；总字数、章节数、每章字数必填</small></div></li>
            <li :class="{ done: !!book.mainline.trim() && worldReady() && !!charactersDraft.length }"><span>2</span><div><strong>AI 生成设定</strong><small>单主线 + 世界观 + 人物（只读）</small></div></li>
            <li :class="{ done: !!currentUnit?.chosen_plan }"><span>3</span><div><strong>大事件目录</strong><small>随三套方案一次生成，采用即选定</small></div></li>
            <li :class="{ done: routeChosen(currentUnit) }"><span>4</span><div><strong>章节方案与路线</strong><small>一键并行拆分，逐事件选路线</small></div></li>
            <li :class="{ done: !!((currentUnit?.merged_route || {}).chapters || []).length }"><span>5</span><div><strong>校验合并</strong><small>AI 查重补缺，给出合并方案</small></div></li>
            <li :class="{ done: !!((currentUnit?.merged_route || {}).chapters || []).length && (currentUnit?.merged_route?.chapters || []).every(chapter => chapter.draft) }"><span>6</span><div><strong>串行写正文</strong><small>每章带上前文全部章节，逐章写完</small></div></li>
          </ol>
        </div>

        <div class="card">
          <div class="card-title">
            <div class="card-title-inner"><span class="step-badge">1</span><div><h2>参考文档与全书规模</h2><p>上传 .txt / .md / .epub 文件（可跳过），填好总字数、大概章节数与每章大概字数；AI 会按这个规模合并路线，避免章节拆得太碎。</p></div></div>
            <span class="status-chip" :class="{ done: bookScaleReady() }">{{ bookScaleReady() ? '规模已定' : '规模必填' }}</span>
            <button class="primary-button" @click="saveBookMeta">保存全书规模</button>
          </div>
          <div class="two world-fields">
            <label>
              <span>大概总字数（必填）</span>
              <input v-model.number="book.target_words" class="hero-input" type="number" min="1" step="1000" placeholder="例如：90000" />
            </label>
            <label>
              <span>大概章节数（必填）</span>
              <input v-model.number="book.target_chapters" class="hero-input" type="number" min="1" step="1" placeholder="例如：36" />
            </label>
            <label>
              <span>每章大概字数（必填）</span>
              <input v-model.number="book.words_per_chapter" class="hero-input" type="number" min="500" step="500" placeholder="例如：2500" />
            </label>
          </div>
          <p class="scale-note muted">先保存「总字数 / 章节数 / 每章字数」这组规模再让 AI 生成；合并路线会严格按这个规模收拢子事件（一章可以包含多个子事件）。</p>
          <div class="upload-row">
            <label class="primary-button file-button">上传文档<input type="file" accept=".txt,.md,.epub,text/plain" class="file-input" @change="uploadReference" /></label>
            <span class="muted">没有参考文档也可以直接输入想法让 AI 生成设定。</span>
          </div>
          <div v-if="book.references?.length" class="reference-list">
            <div v-for="ref in book.references" :key="ref.name" class="reference-row">
              <label class="reference-toggle">
                <input type="checkbox" :checked="ref.enabled" @change="toggleReference(ref.name, $event.target.checked)" />
                <span>{{ ref.name }}</span>
              </label>
              <small>{{ ref.chars }} 字 · {{ formatBytes(ref.size) }}</small>
              <button class="icon" title="删除" @click="deleteReference(ref.name)">×</button>
            </div>
          </div>
          <div class="reference-limit-row">
            <label>
              <span>发送给 AI 的参考长度</span>
              <select v-model.number="book.reference_limit" @change="saveReferenceLimit">
                <option :value="0">全文（不截断）</option>
                <option :value="20000">节选 2 万字符</option>
                <option :value="40000">节选 4 万字符</option>
                <option :value="80000">节选 8 万字符</option>
                <option :value="12000">节选 1.2 万字符</option>
              </select>
            </label>
            <small>0 = 全文；越大越完整，但也越耗 token、占用更多上下文。</small>
          </div>
          <div id="proposals-section" class="reference-brief">
            <div class="reference-brief-head">
              <strong>一键生成完整全书路线（LangGraph 全流程）</strong>
              <small>一个按钮跑完全部 8 个阶段：并行生成三套方案 → 三个 Agent 立论攻辩 → 主持人裁决 → 自动采用获胜方案 → 逐事件拆解评分 → 校验合并并给每章命名 → 串行撰写各章正文（每章都带上前面的完整章节）。</small>
            </div>
            <textarea v-model="referencePlan" rows="3" placeholder="告诉 AI 你想写一部什么样的书：题材、标签、关系走向、想要的氛围…"></textarea>
            <div class="reference-brief-actions">
              <button class="ai-button" :disabled="!referencePlan.trim() || !bookScaleReady() || pipelineJob.active" @click="startPipeline">{{ pipelineJob.active ? 'LangGraph 全流程运行中…' : '一键生成完整全书路线' }}</button>
              <small v-if="bookScaleReady()" class="muted">已带上规模：约 {{ book.target_words }} 字 / {{ book.target_chapters }} 章 / 每章约 {{ book.words_per_chapter }} 字；生成、辩论、采用、拆分、合并、写正文全自动。</small>
              <small v-else class="muted warn-text">还没保存完整规模（总字数 / 章节数 / 每章字数），本按钮暂时不可用，请先填好并保存。</small>
            </div>

            <div class="pipeline-vis">
              <template v-for="(node, index) in pipelineNodes" :key="node.key">
                <div v-if="index" class="pipeline-link" :class="pipelineNodeState(node.key)">↓</div>
                <div class="pipeline-node" :class="pipelineNodeState(node.key)">
                  <span class="pipeline-dot"><span v-if="pipelineNodeState(node.key) === 'active'" class="spinner"></span><span v-else-if="pipelineNodeState(node.key) === 'done'">✓</span></span>
                  <div class="pipeline-node-copy">
                    <strong>{{ node.label }}</strong>
                    <small>{{ node.hint }}</small>
                  </div>
                </div>
              </template>
              <p v-if="pipelineJob.active || pipelineJob.stage === 'done'" class="pipeline-progress">{{ pipelineJob.stage_label }}：{{ pipelineJob.detail }}</p>
            </div>

            <div v-if="proposals.length" class="proposal-list">
              <div v-for="proposal in proposals" :key="proposal.key" class="proposal">
                <div class="proposal-head">
                  <div class="proposal-head-info">
                    <strong>方案 {{ proposal.key }} · {{ proposal.title || '未命名' }}</strong>
                    <p v-if="proposal.thesis" class="proposal-thesis">{{ proposal.thesis }}</p>
                  </div>
                </div>
                <details>
                  <summary>展开看主线 / 世界观 / 人物 / 大事件目录</summary>
                  <div class="proposal-body">
                    <div class="proposal-section">
                      <h4>单主线</h4>
                      <p class="proposal-mainline">{{ proposal.mainline }}</p>
                    </div>
                    <div class="proposal-section">
                      <h4>世界观</h4>
                      <p><b>类型</b>：{{ proposal.world?.genre || '—' }}</p>
                      <p><b>前提</b>：{{ proposal.world?.premise || '—' }}</p>
                      <p><b>时代与地点</b>：{{ proposal.world?.era_place || '—' }}</p>
                      <p><b>规则</b>：{{ proposal.world?.rules || '—' }}</p>
                      <p><b>基调</b>：{{ proposal.world?.tone || '—' }}</p>
                    </div>
                    <div class="proposal-section">
                      <h4>人物（{{ proposal.characters?.length || 0 }}）</h4>
                      <div class="proposal-chars">
                        <span v-for="(character, charIndex) in proposal.characters" :key="charIndex" class="proposal-char">
                          <b>{{ character.name || '未命名' }}</b>
                          <small>{{ [character.gender, character.age].filter(Boolean).join(' · ') }}</small>
                          <span>{{ character.notes }}</span>
                        </span>
                      </div>
                    </div>
                    <div class="proposal-section">
                      <h4>大事件目录（{{ proposal.directory?.length || 0 }} 个事件）</h4>
                      <ol class="proposal-steps">
                        <li v-for="step in proposal.directory" :key="step.seq"><b>{{ step.who || '（未写人物）' }}</b> → {{ step.what }}</li>
                      </ol>
                    </div>
                  </div>
                </details>
              </div>
            </div>

            <div class="debate-box">
              <div class="debate-head">
                <div>
                  <strong>辩论结果与主持人裁决</strong>
                  <p class="muted">三个 Agent 的立论、攻辩全程可见；主持人评分排名后，获胜方案会被流水线自动采用，随后拆分、合并、串行写完各章正文，无需手动操作。结果保存在项目里，刷新不丢。</p>
                </div>
              </div>
              <p v-if="!debate && !pipelineJob.active" class="muted">点一次上面的按钮，这里会展示评分排名、立论/攻辩记录和主持人的最终方案。</p>

              <template v-if="debate && proposals.length">
                <div v-if="debate.verdict?.ranking?.length" class="debate-verdict">
                  <div v-for="item in debate.verdict.ranking" :key="item.key" class="ranking-row" :class="{ winner: String(item.key) === String(debate.verdict?.decision) }">
                    <div class="ranking-head"><span class="ranking-pos">方案 {{ item.key }}</span><span class="ranking-score">{{ item.score }} 分</span><span v-if="String(item.key) === String(debate.verdict?.decision)" class="chosen-tag">主持人推荐</span></div>
                    <div class="ranking-body">
                      <p><b>优点</b>：{{ item.strengths }}</p>
                      <p><b>弱点</b>：{{ item.weaknesses }}</p>
                      <p><b>评语</b>：{{ item.comment }}</p>
                    </div>
                  </div>
                  <p v-if="debate.verdict?.summary" class="debate-summary pre-wrap">{{ debate.verdict.summary }}</p>
                </div>

                <div class="debate-log">
                  <details v-for="proposal in proposals" :key="proposal.key">
                    <summary>查看方案 {{ proposal.key }} 的立论与攻辩</summary>
                    <div class="debate-log-body">
                      <h4>立论</h4>
                      <p class="pre-wrap">{{ debate.cases?.[proposal.key] || '（未立论）' }}</p>
                      <template v-for="(text, index) in (debate.rebuttals?.[proposal.key] || [])" :key="index">
                        <h4>第 {{ index + 1 }} 轮攻辩</h4>
                        <p class="pre-wrap">{{ text }}</p>
                      </template>
                    </div>
                  </details>
                </div>

                <div v-if="debate.merged && debate.verdict?.decision === 'M'" class="proposal merged-card">
                  <div class="proposal-head">
                    <div class="proposal-head-info">
                      <strong>主持人融合方案 · {{ debate.merged.title || '未命名' }}</strong>
                      <p v-if="debate.merged.thesis" class="proposal-thesis">{{ debate.merged.thesis }}</p>
                    </div>
                  </div>
                  <details>
                    <summary>展开看主线 / 大事件目录</summary>
                    <div class="proposal-body">
                      <div class="proposal-section">
                        <h4>单主线</h4>
                        <p class="proposal-mainline">{{ debate.merged.mainline }}</p>
                      </div>
                      <div class="proposal-section">
                        <h4>大事件目录（{{ debate.merged.directory?.length || 0 }} 个事件）</h4>
                        <ol class="proposal-steps">
                          <li v-for="step in debate.merged.directory" :key="step.seq"><b>{{ step.who || '（未写人物）' }}</b> → {{ step.what }}</li>
                        </ol>
                      </div>
                    </div>
                  </details>
                </div>
              </template>
            </div>
          </div>
          <div class="two world-fields">
            <label class="wide">
              <span>书名（AI 生成）</span>
              <input v-model="book.title" class="ro-input" placeholder="书名待 AI 生成" disabled />
            </label>
            <label class="wide">
              <span>单主线（AI 生成，只读）</span>
              <textarea v-model="book.mainline" class="ro-area" rows="3" placeholder="全流程跑完后自动回填到这里。" disabled></textarea>
            </label>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <div class="card-title-inner"><span class="step-badge">2</span><div><h2>世界观（只读）</h2><p>由「生成三套全书方案」统一生成，采用后写入。类型只是标签，细节由下面的字段承载。</p></div></div>
          </div>
          <div class="genre-row">
            <span class="genre-tag">{{ book.world.genre || '类型待 AI 生成' }}</span>
          </div>
          <div class="two world-fields">
            <label v-for="field in worldFields" :key="field.key" :class="{ wide: field.key === 'premise' || field.key === 'rules' }">
              <span>{{ field.label }}</span>
              <textarea v-model="book.world[field.key]" class="ro-area" rows="2" :placeholder="field.placeholder" disabled></textarea>
            </label>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <div class="card-title-inner"><span class="step-badge">3</span><div><h2>人物（只读）</h2><p>由 AI 根据参考文档生成的人物库；后续章节拆解只使用这里的人物。</p></div></div>
          </div>
          <div v-if="charactersDraft.length" class="ro-chars">
            <div v-for="(item, index) in charactersDraft" :key="index" class="ro-char">
              <div class="ro-char-head">
                <strong>{{ item.name || '未命名人物' }}</strong>
                <span>{{ [item.gender, item.age].filter(Boolean).join(' · ') || '资料待生成' }}</span>
              </div>
              <p>{{ item.notes || '（暂无介绍）' }}</p>
            </div>
          </div>
          <div v-else class="char-empty">
            <p>人物库还是空的：在上方「生成三套全书方案」并采用后，人物会自动回填到这里。</p>
          </div>
        </div>

        <div class="next-row">
          <p>生成三套方案并采用一套后（主线、世界观、人物、大事件目录全部就位），点右边直接进入下一步：拆章节。</p>
          <button class="primary-button" @click="goToUnits">去下一步：拆章节 ↓</button>
        </div>
      </section>

      <!-- ============ 全文创作 ============ -->
      <section id="units-section" class="units-page">
        <div class="unit-body">
          <template v-if="currentUnit">
            <div class="card flow-bar">
              <div class="flow-bar-head">
                <div class="flow-bar-title">
                  <h2>{{ book.title || currentUnit.title || '一本书' }}</h2>
                  <p>{{ unitProgress(currentUnit) }}% 完成 · {{ unitStatus(currentUnit) }}</p>
                </div>
                <div class="flow-bar-side">
                  <span class="flow-stat">大事件 <b>{{ currentUnit?.segments?.length || 0 }}</b></span>
                  <span class="flow-stat">章节 <b>{{ currentUnit ? eventChapters(currentUnit).length : 0 }}</b></span>
                  <span class="flow-stat">已写 <b>{{ draftWordCount() }}</b> 字</span>
                  <button class="ghost sm danger-text" @click="deleteUnit(currentUnit.id)">重置工作区</button>
                </div>
              </div>
              <ol>
                <li v-for="(flow, index) in currentFlow" :key="flow.key" :class="flow.state">
                  <span class="flow-num">{{ flow.done ? '✓' : index + 1 }}</span>
                  <div><strong>{{ flow.label }}</strong><small>{{ flow.hint }}</small></div>
                </li>
              </ol>
            </div>

            <div class="card step-card" data-card="directory">
              <div class="step-card-head">
                <span class="step-badge">4</span>
                <div class="step-card-title"><h2>大事件目录</h2><p>大事件目录和主线、世界观、人物是同一套方案：在上方「生成三套全书方案」并采用后，这里直接显示采用那套的目录，不用再单独生成。</p></div>
                <span class="status-chip" :class="{ done: !!currentUnit.chosen_plan }">{{ currentUnit.chosen_plan ? '已选定' : '等待生成' }}</span>
              </div>
              <div class="step-card-body">
                <label>拆分与合并的补充要求（可选）<input v-model="form.guidance" placeholder="例如：事件之间要有因果推进" /></label>
                <div v-if="!currentUnit.directory_options?.length" class="ai-action-row">
                  <button class="ghost" @click="goToProposals">还没有大事件目录：去上方生成三套方案 ↑</button>
                  <p class="muted">三套方案里每套都自带大事件目录，采用哪套就带出哪套。</p>
                </div>
                <div v-if="!settingsReady()" class="settings-fix">
                  <div v-if="!bookScaleReady()" class="scale-fix-row">
                    <span class="muted">还差「全书规模」，在这里补上就能解锁：</span>
                    <input v-model.number="book.target_words" class="hero-input" type="number" min="1000" step="1000" placeholder="总字数，例如 90000" />
                    <input v-model.number="book.target_chapters" class="hero-input" type="number" min="1" step="1" placeholder="章节数，例如 36" />
                    <input v-model.number="book.words_per_chapter" class="hero-input" type="number" min="500" step="500" placeholder="每章字数，例如 2500" />
                    <button class="primary-button sm" :disabled="!bookScaleReady()" @click="saveBookMeta">保存全书规模</button>
                  </div>
                  <p v-else class="muted">设定还没就绪：回到上方「生成三套全书方案」并采用一套，主线、世界观、人物和大事件目录会一次写入。</p>
                </div>
                <div v-if="currentUnit.directory_options?.length" class="plan-list">
                  <div v-for="plan in currentUnit.directory_options" :key="plan.key" class="plan" :class="{ chosen: currentUnit.chosen_plan === plan.key }">
                    <div class="plan-head">
                      <strong>方案 {{ plan.key }} · {{ plan.title || '未命名' }}</strong>
                      <span v-if="plan.angle" class="angle-tag">角度：{{ plan.angle }}</span>
                      <button v-if="currentUnit.chosen_plan !== plan.key" class="ghost" @click="choosePlan(plan.key)">采用这套</button>
                      <span v-else class="chosen-tag">当前大事件目录</span>
                    </div>
                    <p v-if="plan.thesis" class="plan-thesis" data-kind="plan-thesis" :data-plan-key="plan.key">{{ plan.thesis }}</p>
                    <ol v-if="plan.steps?.length">
                      <li v-for="step in plan.steps" :key="step.seq" data-kind="plan-step" :data-plan-key="plan.key" :data-step-seq="step.seq"><b>{{ step.who || '（未写人物）' }}</b> → {{ step.what || '（未写动作）' }}</li>
                    </ol>
                    <p v-else class="muted">{{ plan.thesis }}</p>
                  </div>
                </div>
              </div>
            </div>

            <div class="card step-card" data-card="segments">
              <div class="step-card-head">
                <span class="step-badge">5</span>
                <div class="step-card-title"><h2>章节方案与路线选择</h2><p>选定大事件目录后，一键并行生成所有事件的章节方案，在连线圈里逐事件选定路线，再让 AI 校验合并成一条完整路线。</p></div>
                <span class="status-chip" :class="{ done: routeChosen(currentUnit) }">{{ routeChosen(currentUnit) ? '路线已定' : '待选择' }}</span>
              </div>
              <div class="step-card-body">
                <div v-if="!currentUnit.chosen_plan" class="notice-line muted">请先在上方生成三套全书方案并采用一套，大事件目录选定后这里才会列出事件。</div>
                <div v-else-if="!currentUnit.segments?.length" class="notice-line muted">大事件目录已就绪，正在读取事件…</div>
                <template v-else>
                  <div class="batch-row">
                    <button class="primary-button" :disabled="batchJob.active" @click="batchGenerateChapters(false)">{{ batchJob.active ? '正在并行生成…' : '一键并行生成所有事件的章节方案' }}</button>
                    <button class="primary-button" :disabled="batchJob.active" @click="batchGenerateChapters(false, true)">{{ batchJob.active ? '正在并行生成…' : '自动：拆解并按评分选路线' }}</button>
                    <button v-if="currentUnit.segments.some(event => event.chapter_options?.length) && !batchJob.active" class="ghost sm" @click="batchGenerateChapters(true)">重新生成全部</button>
                    <span v-if="batchJob.active" class="batch-progress">{{ batchJob.detail || '排队中…' }}（{{ batchJob.done }}/{{ batchJob.total }}）</span>
                    <span v-else-if="batchJob.message" class="batch-progress warn">{{ batchJob.message }}</span>
                    <p v-else class="muted">每个事件并行生成三套方案并打分；「自动」按钮会按评分选定路线，然后自动校验合并并给章节起名。</p>
                  </div>

                  <div v-if="chapterOptionsReady()" class="route-flow">
                    <div class="route-flow-head">
                      <div class="route-flow-title">路线选择：从上往下，逐一为每个事件挑一套章节拆分（卡片带评分）</div>
                      <div class="muted">每一列是一套方案，卡片里能看到它拆出的全部章节、字数占比与 AI 评分；点击卡片即采用。选中同列的相邻事件会连成一条路径；全部选好后可一键校验合并。</div>
                    </div>
                    <div class="route-grid-scroll">
                      <div class="route-grid" :style="{ gridTemplateColumns: `minmax(150px, 0.8fr) repeat(${routeGrid.keys.length || 1}, minmax(0, 1fr))` }">
                        <div class="route-grid-corner"></div>
                        <div v-for="key in routeGrid.keys" :key="'head-' + key" class="route-grid-head">方案 {{ key }}</div>
                        <template v-for="(event, eventIndex) in routeGrid.rows" :key="'row-' + event.seq">
                          <div class="route-grid-event">
                            <span class="unit-seq">事件 {{ event.seq }}</span>
                            <div class="route-grid-event-what" :title="`${event.who || ''} → ${event.what || ''}`"><b>{{ event.who || '（未写人物）' }}</b> <span class="route-grid-event-arrow">→</span> {{ event.what || '（未写动作）' }}</div>
                          </div>
                          <div v-for="key in routeGrid.keys" :key="`cell-${event.seq}-${key}`" class="route-grid-cell">
                            <div v-if="routePlanOf(event, key)" class="route-card" :class="{ chosen: String(event.chosen_chapter_plan) === String(key) }" @click="chooseEventChapterPlan(event.seq, key)">
                              <div class="route-card-head">
                                <strong>{{ routePlanOf(event, key).title || '未命名' }}</strong>
                                <span v-if="routePlanOf(event, key).score" class="score-tag" :title="routePlanOf(event, key).score_reason">{{ routePlanOf(event, key).score }} 分</span>
                                <span v-if="String(event.chosen_chapter_plan) === String(key)" class="chosen-tag">已选</span>
                              </div>
                              <p v-if="routePlanOf(event, key).thesis" class="route-card-thesis">{{ routePlanOf(event, key).thesis }}</p>
                              <ol class="route-card-steps">
                                <li v-for="step in routePlanOf(event, key).steps" :key="step.seq">
                                  <span class="route-card-seq">第 {{ step.seq }} 章</span>
                                  <span class="route-card-text"><b>{{ step.who || '（未写人物）' }}</b> → {{ step.what || '（未写动作）' }}</span>
                                  <small v-if="step.word_estimate || step.word_ratio">约 {{ step.word_estimate }} 字 · 占全书 {{ step.word_ratio }}%</small>
                                </li>
                              </ol>
                              <div class="route-card-foot">{{ (routePlanOf(event, key).steps || []).length }} 章 · 预计占全书 {{ planWordRatioSum(routePlanOf(event, key)) }}%</div>
                            </div>
                            <div v-else class="route-grid-empty">该事件无此方案</div>
                          </div>
                          <template v-if="eventIndex < routeGrid.rows.length - 1">
                            <div class="route-grid-corner"></div>
                            <div v-for="key in routeGrid.keys" :key="`link-${event.seq}-${key}`" class="route-link-cell">
                              <span class="route-link" :class="{ linked: routeLinked(routeGrid.rows[eventIndex], routeGrid.rows[eventIndex + 1], key) }">↓</span>
                            </div>
                          </template>
                        </template>
                      </div>
                    </div>
                  </div>

                  <div class="route-merge">
                    <div class="merge-bar">
                      <template v-if="routeChosen(currentUnit)">
                        <button class="ai-button merge-button" :disabled="merging" @click="mergeRoute">
                          <span v-if="merging" class="spinner"></span>
                          {{ merging ? '正在校验合并（约 1 分钟）…' : '让 AI 校验并合并整条路线' }}
                        </button>
                        <p class="muted">检查重复与缺口，补充修正后给出唯一一条合并方案。</p>
                      </template>
                      <p v-else class="muted">↑ 每个事件都选好方案后，才能让 AI 校验去重、补齐缺口，合并成一条完整路线。</p>
                    </div>

                    <div v-if="currentUnit.merged_route" class="merged-route">
                      <div class="merged-route-head">
                        <div>
                          <span class="unit-seq">合并路线</span>
                          <h3>{{ currentUnit.merged_route.title || '未命名' }}</h3>
                          <p v-if="currentUnit.merged_route.summary" class="merged-summary">{{ currentUnit.merged_route.summary }}</p>
                        </div>
                        <button class="ghost sm merge-button" :disabled="merging" @click="mergeRoute">
                          <span v-if="merging" class="spinner"></span>
                          {{ merging ? '正在合并…' : '重新校验合并' }}
                        </button>
                      </div>
                      <p v-if="!currentUnit.merged_route.chapters?.length" class="muted">模型没有返回结构化章节，可点「重新校验合并」再试一次。</p>
                      <ul v-else-if="currentUnit.merged_route.fixed?.length" class="merged-fix">
                        <li v-for="(item, index) in currentUnit.merged_route.fixed" :key="index">{{ item }}</li>
                      </ul>
                      <ol v-if="currentUnit.merged_route.chapters?.length" class="merged-chapters">
                        <li v-for="chapter in currentUnit.merged_route.chapters" :key="`merged-${chapter.seq}`">
                          <span class="unit-seq">第 {{ chapter.seq }} 章</span>
                          <div class="merged-step">
                            <strong v-if="chapter.title" class="merged-title">{{ chapter.title }}</strong>
                            <span class="merged-line"><b>{{ chapter.who || '（未写人物）' }}</b> <span class="route-grid-event-arrow">→</span> {{ chapter.what || '（未写动作）' }}</span>
                          </div>
                          <small v-if="chapter.word_estimate || chapter.word_ratio" class="step-words">（约 {{ chapter.word_estimate }} 字 · 占全书 {{ chapter.word_ratio }}%）</small>
                          <ul v-if="chapter.sub_events?.length" class="merged-sub-events">
                            <li v-for="(sub, subIndex) in chapter.sub_events" :key="subIndex">
                              <b>{{ sub.who || '（未写人物）' }}</b> → {{ sub.what || '（未写内容）' }}
                              <small v-if="sub.word_estimate">（约 {{ sub.word_estimate }} 字）</small>
                            </li>
                          </ul>
                          <details v-if="chapter.draft" class="merged-draft">
                            <summary>查看第 {{ chapter.seq }} 章正文（{{ chapter.draft.length }} 字）</summary>
                            <p class="pre-wrap">{{ chapter.draft }}</p>
                          </details>
                        </li>
                      </ol>
                      <div v-if="currentUnit.merged_route.chapters?.length" class="route-card-foot">合并后共 {{ currentUnit.merged_route.chapters.length }} 章</div>
                    </div>
                  </div>

                </template>
              </div>
            </div>
          </template>
          <div v-else class="card empty-cta">
            <h2>开始这本书</h2>
            <p>先在上方完成设定（总字数 + AI 生成的单主线/世界观/人物），再来这里让 AI 出大事件目录，一键拆分章节方案、选定路线。</p>
            <button class="primary-button" @click="createUnit">初始化全书工作区</button>
          </div>
        </div>
      </section>

    </main>

    <aside v-if="showTrace" class="trace-panel" aria-label="实时日志">
      <div class="trace-head">
        <div><h2>实时日志</h2><p>{{ traceStatus }}</p></div>
        <div class="trace-actions">
          <label class="trace-auto"><input v-model="traceAutoScroll" type="checkbox" />自动滚动</label>
          <button class="ghost" @click="clearTrace">清空显示</button>
          <button class="icon" @click="closeTrace">×</button>
        </div>
      </div>
      <div ref="traceBox" class="trace-body">
        <p v-if="!traceEvents.length" class="muted">暂时没有日志。生成过程中这里会实时显示每一步。</p>
        <div v-for="event in traceEvents" :key="event._uid" class="trace-item" :class="`is-${event.type}`">
          <div class="trace-row" @click="toggleTrace(event._uid)">
            <span class="trace-time">{{ event.timestamp?.slice(11) }}</span>
            <span class="trace-tag">{{ TRACE_TYPE_LABEL[event.type] || event.type }}</span>
            <span class="trace-text">
              <strong v-if="event.type === 'request'">{{ event.label }} · {{ event.model }}</strong>
              <template v-else>{{ tracePreview(event) }}</template>
            </span>
            <button class="trace-expand" :title="expandedTraces.has(event._uid) ? '收起' : '查看完整内容'" @click.stop="toggleTrace(event._uid)">{{ expandedTraces.has(event._uid) ? '−' : '＋' }}</button>
          </div>
          <div v-if="expandedTraces.has(event._uid)" class="trace-detail">
            <pre>{{ event.content }}</pre>
          </div>
        </div>
      </div>
    </aside>

    <div v-if="sel.visible" class="ask-toolbar" :style="{ left: sel.x + 'px', top: sel.y + 'px' }" @mousedown.prevent>
      <button @click="openAsk()">自己问</button>
      <button class="ask-toolbar-close" title="关闭" @click="sel.visible = false">×</button>
    </div>

    <div v-if="ask.show" class="modal-backdrop" @click.self="closeAsk">
      <div class="modal ask-modal">
        <div class="modal-title">
          <div><h2>问 AI</h2><p>围绕选中的文字连续提问，历史会自动带上</p></div>
          <button class="icon" @click="closeAsk">×</button>
        </div>
        <div class="ask-body">
          <div class="ask-selection"><h3>选中的文字</h3><pre>{{ ask.text }}</pre></div>

          <div v-if="!ask.messages.length" class="ask-form">
            <label>你想问 / 想改什么<textarea v-model="ask.question" rows="3" placeholder="例如：这步为什么这样安排？想改的话可以直接说“把这段改成…”"></textarea></label>
            <button class="primary-button" :disabled="ask.busy" @click="submitAsk">{{ ask.busy ? '正在问 AI…' : '发送' }}</button>
          </div>

          <div v-else class="ask-chat">
            <div v-for="(msg, index) in ask.messages" :key="index" class="ask-bubble" :class="msg.role">
              <span class="ask-role">{{ msg.role === 'user' ? '你' : 'AI' }}</span>
              <pre>{{ msg.content }}</pre>
            </div>
            <div v-if="ask.busy" class="ask-bubble assistant"><span class="ask-role">AI</span><pre>正在想…</pre></div>
          </div>

          <div v-if="ask.messages.length" class="ask-input-row">
            <textarea v-model="ask.input" rows="2" placeholder="继续追问，或让它改一下…" @keydown.ctrl.enter="submitAsk" @keydown.meta.enter="submitAsk"></textarea>
            <button class="primary-button" :disabled="ask.busy" @click="submitAsk">发送</button>
          </div>

          <div v-if="ask.result" class="ask-result">
            <div v-if="ask.usedRewrite" class="ask-tool-note">AI 已调用 <code>rewrite_text</code> 工具，正在等待你确认后应用到原文。</div>
            <div class="ask-result-actions">
              <button class="ghost" @click="copyAskResult">复制最近结果</button>
              <button v-if="canApplyAsk" class="primary-button" @click="applyAskResult()">应用到原文</button>
              <button class="ghost" @click="clearAsk">清空对话</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="ask.showRewriteConfirm" class="modal-backdrop rewrite-confirm-layer" @click.self="cancelRewriteConfirm">
      <div class="modal rewrite-confirm-modal">
        <div class="modal-title">
          <div><h2>确认应用改写</h2><p>AI 已调用 rewrite_text，只有你确认后才会写入原文</p></div>
          <button class="icon" @click="cancelRewriteConfirm">×</button>
        </div>
        <div v-if="ask.pendingRewrite?.reason" class="rewrite-section">
          <h3>修改理由</h3>
          <pre>{{ ask.pendingRewrite.reason }}</pre>
        </div>
        <div class="rewrite-section">
          <h3>替换后文本</h3>
          <pre>{{ ask.pendingRewrite?.replacement }}</pre>
        </div>
        <div v-if="ask.pendingRewriteError" class="banner error">{{ ask.pendingRewriteError }}</div>
        <div class="modal-actions">
          <button class="ghost" @click="cancelRewriteConfirm">取消</button>
          <button class="primary-button" @click="confirmRewriteApply">确认应用到原文</button>
        </div>
      </div>
    </div>

    <div v-if="promptPreview.show" class="modal-backdrop prompt-preview-layer" @click.self="closePromptPreview">
      <div class="modal prompt-modal">
        <div class="modal-title">
          <div><h2>{{ promptPreview.title }}</h2><p>{{ promptPreview.label || '这是即将发给模型的原始请求' }}</p></div>
          <button class="icon" :disabled="promptPreview.sending" @click="closePromptPreview">×</button>
        </div>
        <p class="muted">下面是最终发送给模型接口的原始 HTTP 请求体。点“发送请求”才会真正调用模型，点“取消”不会调用。</p>
        <pre class="request-preview">{{ promptPreview.request }}</pre>
        <div v-if="promptPreview.error" class="banner error">{{ promptPreview.error }}</div>
        <div class="modal-actions">
          <button class="ghost" :disabled="promptPreview.sending" @click="closePromptPreview">取消</button>
          <button class="primary-button" :disabled="promptPreview.sending" @click="confirmPromptPreview">{{ promptPreview.sending ? '正在发送…' : '发送请求' }}</button>
        </div>
      </div>
    </div>
  </div>
</template>






















