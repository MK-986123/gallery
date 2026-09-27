// Adapted from llama.cpp commit 60081bb2b5b3294165a4d67c5cbeebe74c868014,
// examples/llama.android/lib/src/main/cpp/ai_chat.cpp (MIT license; see llama.cpp.LICENSE).
#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <unistd.h>
#include <sampling.h>

#include "logging.h"
#include "chat.h"
#include "common.h"
#include "llama.h"

template<class T>
static std::string join(const std::vector<T> &values, const std::string &delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) { str << delim; }
    }
    return str.str();
}

/**
 * LLama resources: context, model, batch and sampler
 */
constexpr int   N_THREADS_MIN           = 2;
constexpr int   N_THREADS_MAX           = 4;
constexpr int   N_THREADS_HEADROOM      = 2;

constexpr int   DEFAULT_CONTEXT_SIZE    = 8192;
constexpr int   OVERFLOW_HEADROOM       = 4;
constexpr int   BATCH_SIZE              = 512;
constexpr float DEFAULT_SAMPLER_TEMP    = 0.3f;

static llama_model                      * g_model;
static llama_context                    * g_context;
static llama_batch                        g_batch;
static common_chat_templates_ptr          g_chat_templates;
static common_sampler                   * g_sampler;
// Context actually allocated for g_context; never larger than the model's training context.
static int                                g_n_ctx = DEFAULT_CONTEXT_SIZE;

// Java strings cross JNI as modified UTF-8, which encodes emoji as surrogate pairs that the
// tokenizer cannot read, so text is exchanged as standard UTF-8 byte arrays instead.
static std::string utf8_from_bytes(JNIEnv *env, jbyteArray bytes) {
    if (!bytes) return {};
    std::string text(env->GetArrayLength(bytes), '\0');
    env->GetByteArrayRegion(bytes, 0, (jsize) text.size(), reinterpret_cast<jbyte *>(text.data()));
    return text;
}

static jbyteArray bytes_from_utf8(JNIEnv *env, const std::string &text) {
    jbyteArray bytes = env->NewByteArray((jsize) text.size());
    if (bytes) {
        env->SetByteArrayRegion(bytes, 0, (jsize) text.size(),
                                reinterpret_cast<const jbyte *>(text.data()));
    }
    return bytes;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_init(JNIEnv *env, jobject /*unused*/, jstring nativeLibDir) {
    // Set llama log handler to Android
    llama_log_set(aichat_android_log_callback, nullptr);

    // Loading all CPU backend variants
    const auto *path_to_backend = env->GetStringUTFChars(nativeLibDir, 0);
    LOGi("Loading backends from %s", path_to_backend);
    ggml_backend_load_all_from_path(path_to_backend);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);

    // Initialize backends
    llama_backend_init();
    LOGi("Backend initiated; Log handler set.");
    return (jint) ggml_backend_dev_count();
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_load(JNIEnv *env, jobject, jstring jmodel_path) {
    llama_model_params model_params = llama_model_default_params();

    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);

    auto *model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!model) {
        return 1;
    }
    g_model = model;
    return 0;
}

static llama_context *init_context(llama_model *model, int n_ctx = DEFAULT_CONTEXT_SIZE) {
    if (!model) {
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    // Multi-threading setup
    const int n_threads = std::max(N_THREADS_MIN, std::min(N_THREADS_MAX,
                                                     (int) sysconf(_SC_NPROCESSORS_ONLN) -
                                                     N_THREADS_HEADROOM));
    LOGi("%s: Using %d threads", __func__, n_threads);

    // Context parameters setup
    llama_context_params ctx_params = llama_context_default_params();
    const int trained_context_size = llama_model_n_ctx_train(model);
    if (trained_context_size > 0 && n_ctx > trained_context_size) {
        // Positions past the training context produce unusable output, so older turns are
        // dropped earlier instead.
        LOGw("%s: Model was trained with only %d context size! Using it instead of %d...",
             __func__, trained_context_size, n_ctx);
        n_ctx = trained_context_size;
    }
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = BATCH_SIZE;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    auto *context = llama_init_from_model(g_model, ctx_params);
    if (context == nullptr) {
        LOGe("%s: llama_new_context_with_model() returned null)", __func__);
    }
    return context;
}

static common_sampler *new_sampler(int top_k, float top_p, float temp) {
    common_params_sampling sparams;
    sparams.top_k = top_k;
    sparams.top_p = top_p;
    sparams.temp = temp;
    return common_sampler_init(g_model, sparams);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_configureSampler(
        JNIEnv *, jobject, jint top_k, jfloat top_p, jfloat temp) try {
    if (!g_model || top_k < 1 || top_k > 100 || top_p < 0.0f || top_p > 1.0f ||
        temp < 0.0f || temp > 2.0f) {
        return 1;
    }
    auto *sampler = new_sampler(top_k, top_p, temp);
    if (!sampler) return 2;
    if (g_sampler) common_sampler_free(g_sampler);
    g_sampler = sampler;
    return 0;
} catch (const std::exception &e) {
    // A C++ exception escaping a JNI call aborts the whole app process.
    LOGe("configureSampler: %s", e.what());
    return 2;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_prepare(JNIEnv * /*env*/, jobject /*unused*/) try {
    auto *context = init_context(g_model);
    if (!context) { return 1; }
    g_context = context;
    g_n_ctx = (int) llama_n_ctx(context);
    g_batch = llama_batch_init(BATCH_SIZE, 0, 1);
    // Throws for chat templates the bundled Jinja parser cannot read.
    g_chat_templates = common_chat_templates_init(g_model, "");
    if (common_chat_templates_was_explicit(g_chat_templates.get())) {
        // Chats are formatted without Jinja; reject templates that path cannot apply at load time
        // instead of on the first message.
        common_chat_templates_inputs probe;
        probe.use_jinja = false;
        probe.messages.emplace_back();
        probe.messages.back().role = "user";
        probe.messages.back().content = "Hi";
        common_chat_templates_apply(g_chat_templates.get(), probe);
    }
    g_sampler = new_sampler(40, 0.95f, DEFAULT_SAMPLER_TEMP);
    return g_chat_templates && g_sampler ? 0 : 2;
} catch (const std::exception &e) {
    LOGe("prepare: %s", e.what());
    return 3;
}

static std::string get_backend() {
    std::vector<std::string> backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *reg = ggml_backend_reg_get(i);
        std::string name = ggml_backend_reg_name(reg);
        if (name != "CPU") {
            backends.push_back(ggml_backend_reg_name(reg));
        }
    }
    return backends.empty() ? "CPU" : join(backends, ",");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_systemInfo(JNIEnv *env, jobject /*unused*/) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_benchModel(JNIEnv *env, jobject /*unused*/, jint pp, jint tg,
                                                      jint pl, jint nr) {
    auto *context = init_context(g_model, pp);
    if (!context) {
        const auto *const err_msg = "Fail to init_context! Bench aborted.";
        LOGe(err_msg);
        return env->NewStringUTF(err_msg);
    }

    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const uint32_t n_ctx = llama_n_ctx(context);
    LOGi("n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp = %d)", pp);

        common_batch_clear(g_batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(g_batch, 0, i, {0}, false);
        }

        g_batch.logits[g_batch.n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, g_batch) != 0) {
            LOGe("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg = %d)", tg);

        llama_memory_clear(llama_get_memory(context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {
            common_batch_clear(g_batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(g_batch, 0, i, {j}, true);
            }

            if (llama_decode(context, g_batch) != 0) {
                LOGe("llama_decode() failed during text generation");
            }
        }
        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    llama_free(context);

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(g_model, model_desc, sizeof(model_desc));

    const auto model_size = double(llama_model_size(g_model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(g_model)) / 1e9;

    const auto backend = get_backend();
    std::stringstream result;
    result << std::setprecision(3);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";
    return env->NewStringUTF(result.str().c_str());
}


/**
 * Completion loop's long-term states:
 * - chat management
 * - position tracking
 */
constexpr const char *ROLE_SYSTEM       = "system";
constexpr const char *ROLE_USER         = "user";
constexpr const char *ROLE_ASSISTANT    = "assistant";

static std::vector<common_chat_msg> chat_msgs;
static llama_pos system_prompt_position;
static llama_pos current_position;
// True when the cache ends with a sampled token instead of template-formatted text.
static bool cache_ends_with_sampled_token;

static void reset_long_term_states(const bool clear_kv_cache = true) {
    chat_msgs.clear();
    system_prompt_position = 0;
    current_position = 0;
    cache_ends_with_sampled_token = false;

    if (clear_kv_cache && g_context)
        llama_memory_clear(llama_get_memory(g_context), false);
}

static common_chat_msg make_chat_msg(const std::string &role, const std::string &content) {
    common_chat_msg msg;
    msg.role = role;
    msg.content = content;
    return msg;
}

// Formats the text that follows `past` in the transcript. Generation stops at the end-of-turn
// token, before the separator the template writes after it, so that separator is restored only
// when the cache ends with a sampled token (the same heuristic as common_chat_format_single).
static std::string format_chat_msg(const std::vector<common_chat_msg> &past,
                                   const common_chat_msg &msg,
                                   const bool add_generation_prompt,
                                   const bool after_sampled_token) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = false;
    std::string formatted_past;
    if (!past.empty()) {
        inputs.messages = past;
        inputs.add_generation_prompt = false;
        formatted_past = common_chat_templates_apply(g_chat_templates.get(), inputs).prompt;
    }
    inputs.messages.push_back(msg);
    inputs.add_generation_prompt = add_generation_prompt;
    const auto formatted = common_chat_templates_apply(g_chat_templates.get(), inputs).prompt;
    const bool restore_separator =
            after_sampled_token && !formatted_past.empty() && formatted_past.back() == '\n';
    return (restore_separator ? "\n" : "") + formatted.substr(formatted_past.size());
}

// Only the first text in the sequence may receive BOS; later turns rely on the template's own
// separators, and a BOS in the middle of a conversation degrades BOS-prefixed models.
static llama_tokens tokenize_at(const std::string &text, const llama_pos position,
                                const bool parse_special) {
    return common_tokenize(g_context, text, /* add_special */ position == 0, parse_special);
}

/**
 * Completion loop's short-term states:
 * - stop generation position
 * - token chars caching
 * - current assistant message being generated
 */
static int generated_token_count;
static int max_generated_tokens;
static std::string cached_token_chars;
static std::ostringstream assistant_ss;

static void reset_short_term_states() {
    generated_token_count = 0;
    max_generated_tokens = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
}

static int decode_tokens_in_batches(
        llama_context *context,
        llama_batch &batch,
        const llama_tokens &tokens,
        const llama_pos start_pos,
        const bool compute_last_logit = false) {
    // Process tokens in batches using the global batch
    LOGd("%s: Decode %d tokens starting at position %d", __func__, (int) tokens.size(), start_pos);
    for (int i = 0; i < (int) tokens.size(); i += BATCH_SIZE) {
        const int cur_batch_size = std::min((int) tokens.size() - i, BATCH_SIZE);
        common_batch_clear(batch);
        LOGv("%s: Preparing a batch size of %d starting at: %d", __func__, cur_batch_size, i);

        // Callers reserve enough room before decoding; a partial replay must never shift its
        // system prompt or silently drop an earlier saved turn.
        if (start_pos + i + cur_batch_size >= g_n_ctx - OVERFLOW_HEADROOM) {
            LOGe("%s: Batch exceeds context", __func__);
            return 1;
        }

        // Add tokens to the batch with proper positions
        for (int j = 0; j < cur_batch_size; j++) {
            const llama_token token_id = tokens[i + j];
            const llama_pos position = start_pos + i + j;
            const bool want_logit = compute_last_logit && (i + j == (int) tokens.size() - 1);
            common_batch_add(batch, token_id, position, {0}, want_logit);
        }

        // Decode this batch
        const int decode_result = llama_decode(context, batch);
        if (decode_result) {
            LOGe("%s: llama_decode failed w/ %d", __func__, decode_result);
            return 1;
        }
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processSystemPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jbyteArray jsystem_prompt
) try {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    const std::string system_prompt = utf8_from_bytes(env, jsystem_prompt);
    std::string formatted_system_prompt(system_prompt);

    // Format system prompt if applicable. An empty prompt adds no system turn, matching what the
    // model's own template does without one.
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template && !system_prompt.empty()) {
        const auto msg = make_chat_msg(ROLE_SYSTEM, system_prompt);
        formatted_system_prompt = format_chat_msg(chat_msgs, msg, false, false);
        chat_msgs.push_back(msg);
    }

    // Tokenize system prompt
    const auto system_tokens = tokenize_at(formatted_system_prompt, current_position,
                                           has_chat_template);

    // Handle context overflow
    const int max_batch_size = g_n_ctx - OVERFLOW_HEADROOM;
    if ((int) system_tokens.size() > max_batch_size) {
        LOGe("%s: System prompt too long for context! %d tokens, max: %d",
             __func__, (int) system_tokens.size(), max_batch_size);
        return 1;
    }

    // Decode system tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    system_prompt_position = current_position = (int) system_tokens.size();
    return 0;
} catch (const std::exception &e) {
    LOGe("processSystemPrompt: %s", e.what());
    return 3;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jbyteArray juser_prompt,
        jint n_predict
) try {
    // Reset short-term states
    reset_short_term_states();

    const std::string user_prompt = utf8_from_bytes(env, juser_prompt);
    std::string formatted_user_prompt(user_prompt);

    // Format user prompt if applicable
    const auto msg = make_chat_msg(ROLE_USER, user_prompt);
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_user_prompt =
                format_chat_msg(chat_msgs, msg, true, cache_ends_with_sampled_token);
    }

    // Decode formatted user prompts
    auto user_tokens = tokenize_at(formatted_user_prompt, current_position, has_chat_template);

    // Reject before touching the chat state so a rejected prompt leaves the context unchanged.
    const int user_prompt_size = (int) user_tokens.size();
    if (current_position + user_prompt_size >= g_n_ctx - OVERFLOW_HEADROOM) {
        return 1;
    }

    // Decode user tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    chat_msgs.push_back(msg);
    current_position += user_prompt_size;
    cache_ends_with_sampled_token = false;
    max_generated_tokens = std::max(1, (int) n_predict);
    return 0;
} catch (const std::exception &e) {
    LOGe("processUserPrompt: %s", e.what());
    return 3;
}

// Rebuild a saved text conversation without sampling new answers. The return value is the number
// of older turns omitted to leave room for the next reply, or a negative error code.
extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_replayTurns(
        JNIEnv *env, jobject, jobjectArray jroles, jobjectArray jtexts, jint reserve_tokens) try {
    if (!g_context || !jroles || !jtexts) return -1;
    const jsize count = env->GetArrayLength(jroles);
    if (count != env->GetArrayLength(jtexts) || count > 2000) return -1;

    std::vector<common_chat_msg> turns;
    turns.reserve(count);
    for (jsize i = 0; i < count; ++i) {
        auto role_ref = (jstring) env->GetObjectArrayElement(jroles, i);
        auto text_ref = (jbyteArray) env->GetObjectArrayElement(jtexts, i);
        if (!role_ref || !text_ref) return -1;
        const char *role = env->GetStringUTFChars(role_ref, nullptr);
        if (!role) return -1;
        common_chat_msg turn = make_chat_msg(role, utf8_from_bytes(env, text_ref));
        env->ReleaseStringUTFChars(role_ref, role);
        env->DeleteLocalRef(role_ref);
        env->DeleteLocalRef(text_ref);
        if (turn.role != ROLE_USER && turn.role != ROLE_ASSISTANT) return -1;
        turns.push_back(std::move(turn));
    }

    const bool has_template = common_chat_templates_was_explicit(g_chat_templates.get());
    const int reserve = std::min(std::max(256, std::min((int) reserve_tokens, 2048)), g_n_ctx / 2);
    const int limit = g_n_ctx - OVERFLOW_HEADROOM - reserve;
    // Saved turns are decoded exactly as the template writes a finished transcript. Without a
    // template, turns are concatenated the same way a live prompt and its reply are.
    auto formatted_tokens = [&](const std::vector<common_chat_msg> &history,
                                const common_chat_msg &turn,
                                const llama_pos position,
                                const bool after_sampled_token) {
        const std::string formatted = has_template
            ? format_chat_msg(history, turn, false, after_sampled_token)
            : turn.content;
        return tokenize_at(formatted, position, has_template);
    };
    auto fits = [&](int first) {
        std::vector<common_chat_msg> history = chat_msgs;
        int position = current_position;
        for (int i = first; i < count; ++i) {
            const bool after_sampled_token = i == first && cache_ends_with_sampled_token;
            position += (int) formatted_tokens(history, turns[i], position, after_sampled_token).size();
            if (position > limit) return false;
            history.push_back(turns[i]);
        }
        return true;
    };

    int first = 0;
    while (first < count && turns[first].role != ROLE_USER) ++first;
    while (first < count && !fits(first)) {
        ++first;
        while (first < count && turns[first].role != ROLE_USER) ++first;
    }
    for (int i = first; i < count; ++i) {
        auto tokens = formatted_tokens(chat_msgs, turns[i], current_position,
                                       cache_ends_with_sampled_token);
        if (decode_tokens_in_batches(g_context, g_batch, tokens, current_position) != 0) {
            return -2;
        }
        current_position += (int) tokens.size();
        cache_ends_with_sampled_token = false;
        chat_msgs.push_back(turns[i]);
    }
    return first;
} catch (const std::exception &e) {
    LOGe("replayTurns: %s", e.what());
    return -3;
}

static bool is_valid_utf8(const char *string) {
    if (!string) { return true; }

    const auto *bytes = (const unsigned char *) string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_generateNextToken(
        JNIEnv *env,
        jobject /*unused*/
) try {
    // A reply may end at the context boundary. The next request rebuilds from complete turns.
    if (current_position >= g_n_ctx - OVERFLOW_HEADROOM) {
        chat_msgs.push_back(make_chat_msg(ROLE_ASSISTANT, assistant_ss.str()));
        return nullptr;
    }

    if (generated_token_count >= max_generated_tokens) {
        chat_msgs.push_back(make_chat_msg(ROLE_ASSISTANT, assistant_ss.str()));
        return nullptr;
    }

    // Sample next token
    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);

    // Populate the batch with new token, then decode
    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_position, {0}, true);
    if (llama_decode(g_context, g_batch) != 0) {
        LOGe("%s: llama_decode() failed for generated token", __func__);
        jclass error_class = env->FindClass("java/lang/IllegalStateException");
        if (error_class) env->ThrowNew(error_class, "llama.cpp failed to decode a generated token");
        return nullptr;
    }

    // Update position
    current_position++;
    generated_token_count++;
    cache_ends_with_sampled_token = true;

    // Stop if next token is EOG
    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id)) {
        LOGd("id: %d,\tIS EOG!\nSTOP.", new_token_id);
        chat_msgs.push_back(make_chat_msg(ROLE_ASSISTANT, assistant_ss.str()));
        return nullptr;
    }

    // If not EOG, convert to text
    auto new_token_chars = common_token_to_piece(g_context, new_token_id);
    cached_token_chars += new_token_chars;

    // Return only complete UTF-8 sequences; an empty array means more bytes are pending.
    jbyteArray result = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        result = bytes_from_utf8(env, cached_token_chars);

        assistant_ss << cached_token_chars;
        cached_token_chars.clear();
    } else {
        LOGv("id: %d,\tappend to cache", new_token_id);
        result = bytes_from_utf8(env, "");
    }
    return result;
} catch (const std::exception &e) {
    LOGe("generateNextToken: %s", e.what());
    jclass error_class = env->FindClass("java/lang/IllegalStateException");
    if (error_class) env->ThrowNew(error_class, "llama.cpp failed to generate a token");
    return nullptr;
}


extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_unload(JNIEnv * /*unused*/, jobject /*unused*/) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Free up resources
    if (g_sampler) common_sampler_free(g_sampler);
    g_sampler = nullptr;
    g_chat_templates.reset();
    if (g_context) {
        llama_batch_free(g_batch);
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_shutdown(JNIEnv *, jobject /*unused*/) {
    llama_backend_free();
}
