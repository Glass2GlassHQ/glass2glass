#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "g2g.hpp"

namespace {

constexpr size_t frame_bytes = 2 * 2 * 4;
constexpr int frame_count = 3;
constexpr uint64_t frame_period_ns = 33'333'333;
constexpr uint64_t edited_frame_count = 90;

void expect(bool condition, const std::string &what) {
  if (!condition) {
    throw std::runtime_error("expected: " + what);
  }
}

void round_trip() {
  g2g::AppSrc source("in");
  g2g::AppSink sink("out");
  g2g::Pipeline pipeline("appsrc channel=in caps=video/x-raw,format=RGBA,"
                         "width=2,height=2,framerate=30/1 ! appsink channel=out");

  std::vector<uint8_t> frame(frame_bytes);
  for (int index = 0; index < frame_count; ++index) {
    frame.assign(frame_bytes, static_cast<uint8_t>(index + 1));
    while (!source.try_push(frame.data(), frame.size(), index * frame_period_ns)) {
      std::this_thread::yield();
    }
  }
  source.end_of_stream();

  int pulled = 0;
  while (auto sample = sink.pull()) {
    std::cout << "frame " << pulled << ": " << sample->size() << " bytes, first="
              << int(sample->data()[0]) << ", pts=" << sample->pts_ns() << " ns\n";
    expect(sample->data()[0] == pulled + 1,
           "frame " + std::to_string(pulled) + " carries the bytes pushed for it");
    ++pulled;
  }

  while (auto message = pipeline.poll_bus()) {
    std::cout << "bus: kind=" << int(message->kind) << " " << message->text.value_or("") << "\n";
  }

  G2gStats stats = pipeline.wait();
  std::cout << "done: emitted=" << stats.frames_emitted << " consumed=" << stats.frames_consumed
            << " dropped=" << stats.frames_dropped << "\n";
  expect(pulled == frame_count, "pulled " + std::to_string(pulled) + " of " +
                                    std::to_string(frame_count) + " frames");
}

void edits() {
  g2g::Pipeline pipeline("videotestsrc name=src num-buffers=" +
                         std::to_string(edited_frame_count) +
                         " ! clocksync ! tee name=t ! fakesink");

  std::string valve = pipeline.insert_after("src", "valve drop=false");
  pipeline.set_property(valve, "drop-mode", "forward-sticky-events");
  expect(pipeline.get_property(valve, "drop-mode") == "forward-sticky-events",
         "drop-mode reads back");
  expect(!pipeline.get_property(valve, "no-such-property"),
         "an unknown property reads as empty");
  pipeline.remove(valve);

  std::vector<std::string> branch = pipeline.add_branch("t", "identity ! fakesink");
  expect(branch.size() == 2, "the branch has two names");
  pipeline.remove_branch(branch[1]);

  bool refused = false;
  try {
    pipeline.insert_after("no-such-node", "valve");
  } catch (const g2g::Error &error) {
    std::cout << "refused: " << error.what() << "\n";
    refused = std::string(error.what()).find("UnknownNode") != std::string::npos;
  }
  expect(refused, "inserting after an unknown node is refused with UnknownNode");

  G2gStats stats = pipeline.wait();
  std::cout << "edited run: consumed=" << stats.frames_consumed << "\n";
  expect(stats.frames_consumed == edited_frame_count, "no frame is lost to the edits");
}

} // namespace

int main() {
  round_trip();
  edits();
}
