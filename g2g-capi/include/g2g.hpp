#ifndef G2G_HPP
#define G2G_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "g2g.h"

namespace g2g {

class Error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

enum class BusKind : int {
  StreamStart = G2G_BUS_STREAM_START,
  Eos = G2G_BUS_EOS,
  Error = G2G_BUS_ERROR,
  Warning = G2G_BUS_WARNING,
  Info = G2G_BUS_INFO,
  StateChanged = G2G_BUS_STATE_CHANGED,
  Buffering = G2G_BUS_BUFFERING,
  DurationChanged = G2G_BUS_DURATION_CHANGED,
  Other = G2G_BUS_OTHER,
};

/* `a` and `b` mean what g2g.h documents for G2gBusMessage of the same kind. */
struct BusMessage {
  BusKind kind;
  std::optional<std::string> text;
  uint64_t a;
  uint64_t b;
};

namespace detail {

struct StringFree {
  void operator()(char *text) const { g2g_string_free(text); }
};
struct NamesFree {
  void operator()(char **names) const { g2g_strv_free(names); }
};
struct PipelineFree {
  void operator()(G2gPipeline *pipeline) const { g2g_pipeline_free(pipeline); }
};
struct AppSrcFree {
  void operator()(G2gAppSrc *source) const { g2g_appsrc_free(source); }
};
struct AppSinkFree {
  void operator()(G2gAppSink *sink) const { g2g_appsink_free(sink); }
};
struct SampleFree {
  void operator()(G2gSample *sample) const { g2g_sample_free(sample); }
};

inline std::string take_string(char *text) {
  std::unique_ptr<char, StringFree> owned(text);
  return std::string(owned.get());
}

[[noreturn]] inline void throw_error(char *error) {
  if (error == nullptr) {
    throw Error("g2g refused the call without a reason");
  }
  throw Error(take_string(error));
}

template <typename Edit> std::string take_name(Edit edit) {
  char *error = nullptr;
  char *name = edit(&error);
  if (name == nullptr) {
    throw_error(error);
  }
  return take_string(name);
}

template <typename Edit> std::vector<std::string> take_names(Edit edit) {
  char *error = nullptr;
  std::unique_ptr<char *, NamesFree> names(edit(&error));
  if (!names) {
    throw_error(error);
  }
  std::vector<std::string> owned;
  for (char **entry = names.get(); *entry != nullptr; ++entry) {
    owned.emplace_back(*entry);
  }
  return owned;
}

template <typename Edit> void throw_if_refused(Edit edit) {
  char *error = nullptr;
  if (edit(&error) == 0) {
    throw_error(error);
  }
}

} // namespace detail

/* Calls on one pipeline must come from one thread at a time. Destroying it
 * blocks until the pipeline reaches end of stream. */
class Pipeline {
public:
  explicit Pipeline(const std::string &description) {
    char *error = nullptr;
    handle_.reset(g2g_pipeline_launch(description.c_str(), &error));
    if (!handle_) {
      detail::throw_error(error);
    }
  }

  std::optional<BusMessage> poll_bus() {
    G2gBusMessage raw;
    if (g2g_pipeline_bus_poll(handle_.get(), &raw) == 0) {
      return std::nullopt;
    }
    std::optional<std::string> text;
    if (raw.text != nullptr) {
      text = raw.text;
    }
    return BusMessage{static_cast<BusKind>(raw.kind), text, raw.a, raw.b};
  }

  bool is_done() const { return g2g_pipeline_is_done(handle_.get()) == 1; }

  /* The error itself is posted on the bus. */
  G2gStats wait() {
    G2gStats stats;
    if (g2g_pipeline_wait(handle_.get(), &stats) != 0) {
      throw Error("pipeline ended with an error");
    }
    return stats;
  }

  std::string insert_after(const std::string &node, const std::string &description) {
    return detail::take_name([&](char **error) {
      return g2g_pipeline_insert_after(handle_.get(), node.c_str(), description.c_str(), error);
    });
  }

  std::string insert_before(const std::string &node, const std::string &description) {
    return detail::take_name([&](char **error) {
      return g2g_pipeline_insert_before(handle_.get(), node.c_str(), description.c_str(), error);
    });
  }

  std::vector<std::string> insert_chain_after(const std::string &node,
                                              const std::string &description) {
    return detail::take_names([&](char **error) {
      return g2g_pipeline_insert_chain_after(handle_.get(), node.c_str(), description.c_str(),
                                             error);
    });
  }

  std::vector<std::string> insert_chain_before(const std::string &node,
                                               const std::string &description) {
    return detail::take_names([&](char **error) {
      return g2g_pipeline_insert_chain_before(handle_.get(), node.c_str(), description.c_str(),
                                              error);
    });
  }

  void remove(const std::string &node) {
    detail::throw_if_refused(
        [&](char **error) { return g2g_pipeline_remove(handle_.get(), node.c_str(), error); });
  }

  std::vector<std::string> add_branch(const std::string &tee, const std::string &description) {
    return detail::take_names([&](char **error) {
      return g2g_pipeline_add_branch(handle_.get(), tee.c_str(), description.c_str(), error);
    });
  }

  void remove_branch(const std::string &sink) {
    detail::throw_if_refused([&](char **error) {
      return g2g_pipeline_remove_branch(handle_.get(), sink.c_str(), error);
    });
  }

  std::string add_input(const std::string &muxer, const std::string &description) {
    return detail::take_name([&](char **error) {
      return g2g_pipeline_add_input(handle_.get(), muxer.c_str(), description.c_str(), error);
    });
  }

  std::string replace_sink(const std::string &node, const std::string &description) {
    return detail::take_name([&](char **error) {
      return g2g_pipeline_replace_sink(handle_.get(), node.c_str(), description.c_str(), error);
    });
  }

  std::string replace_source(const std::string &node, const std::string &description) {
    return detail::take_name([&](char **error) {
      return g2g_pipeline_replace_source(handle_.get(), node.c_str(), description.c_str(),
                                         error);
    });
  }

  void set_property(const std::string &node, const std::string &name, const std::string &value) {
    detail::throw_if_refused([&](char **error) {
      return g2g_pipeline_set_property(handle_.get(), node.c_str(), name.c_str(), value.c_str(),
                                       error);
    });
  }

  /* Empty when the element has no such property. */
  std::optional<std::string> get_property(const std::string &node, const std::string &name) {
    char *error = nullptr;
    char *value = g2g_pipeline_get_property(handle_.get(), node.c_str(), name.c_str(), &error);
    if (error != nullptr) {
      detail::throw_error(error);
    }
    if (value == nullptr) {
      return std::nullopt;
    }
    return detail::take_string(value);
  }

private:
  std::unique_ptr<G2gPipeline, detail::PipelineFree> handle_;
};

/* Create it before the pipeline that names its channel. */
class AppSrc {
public:
  explicit AppSrc(const std::string &channel = "default")
      : handle_(g2g_appsrc_new(channel.c_str())) {}

  /* False when the feed is full (retry) or the pipeline is gone. */
  bool try_push(const uint8_t *data, size_t size, uint64_t pts_ns) {
    return g2g_appsrc_push(handle_.get(), data, size, pts_ns) == 1;
  }

  bool end_of_stream() { return g2g_appsrc_end_of_stream(handle_.get()) == 1; }

private:
  std::unique_ptr<G2gAppSrc, detail::AppSrcFree> handle_;
};

/* data() points into the pipeline's buffer and is valid while the Sample lives. */
class Sample {
public:
  const uint8_t *data() const { return g2g_sample_data(handle_.get()); }
  size_t size() const { return g2g_sample_len(handle_.get()); }
  uint64_t pts_ns() const { return g2g_sample_pts(handle_.get()); }

private:
  friend class AppSink;
  explicit Sample(G2gSample *sample) : handle_(sample) {}
  std::unique_ptr<G2gSample, detail::SampleFree> handle_;
};

/* Create it before the pipeline that names its channel. */
class AppSink {
public:
  explicit AppSink(const std::string &channel = "default")
      : handle_(g2g_appsink_new(channel.c_str())) {}

  /* Blocks for the next sample, empty once the stream has ended. */
  std::optional<Sample> pull() {
    G2gSample *sample = nullptr;
    if (g2g_appsink_pull(handle_.get(), &sample) == 0) {
      return std::nullopt;
    }
    return Sample(sample);
  }

private:
  std::unique_ptr<G2gAppSink, detail::AppSinkFree> handle_;
};

} // namespace g2g

#endif /* G2G_HPP */
