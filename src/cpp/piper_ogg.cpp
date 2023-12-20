#include <fcntl.h>
#include <stdio.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "json.hpp"
#include "piper.hpp"
#include "minivorbis.h"
#include "ThreadPool.h"

using namespace std;
using json = nlohmann::json;

stringstream convertWavToOgg(piper::Voice &voice, const vector<int16_t> &input);

int main(int argc, char *argv[]) {
  spdlog::set_default_logger(spdlog::stderr_color_mt("piper_ogg"));
  spdlog::set_level(spdlog::level::info);
  piper::PiperConfig piperConfig;

  if (argc < 2) {
    spdlog::error("Need espeak-ng-data path");
    return 1;
  }

  if (!_isatty(_fileno(stdout))) {
    spdlog::info("Setting stdout to binary mode");
    fflush(stdout);
    _setmode(_fileno(stdout), _O_BINARY);
  }

  piperConfig.eSpeakDataPath = string(argv[1]);
  piper::initialize(piperConfig);

  auto thread_pool = make_unique<ThreadPool>(std::thread::hardware_concurrency());
  unordered_map<string, piper::Voice> voice_map;
  int counter = 0;

  while (!cin.eof() && !cin.bad()) {
    string str;
    // cin.clear();
    getline(cin, str, '\n');
    if (!cin.good()) {
      spdlog::error("Error reading request, got \"{}\"", str);
      continue;
    }
    json request = json::parse(str);
    if (!request.contains("id")) {
      spdlog::error("Request doesn't have \"id\" member.");
      continue;
    }
    auto id = request["id"].get<uint32_t>();

    if (!request.contains("modelPath")) {
      spdlog::error("Request doesn't have \"modelPath\" member.");
      continue;
    }
    auto modelPath = request["modelPath"].get<std::string>();

    if (!request.contains("inputText")) {
      spdlog::error("Request doesn't have \"inputText\" member.");
      continue;
    }
    auto inputText = request["inputText"].get<std::string>();

    piper::Voice *voice = nullptr;
    if (auto iter = voice_map.find(modelPath); iter != voice_map.end()) {
      voice = &iter->second;
    } else {
      piper::Voice newVoice;
      auto [new_iter, inserted] = voice_map.emplace(modelPath, move(newVoice));
      voice = &new_iter->second;
      optional<piper::SpeakerId> speakerId;
      loadVoice(piperConfig, modelPath, modelPath + ".json", *voice, speakerId, false);

      if (voice->synthesisConfig.sampleWidth != 2) {
        spdlog::error("Expected sample width to be 2, got {}", voice->synthesisConfig.sampleWidth);
        continue;
      }
    }

    stringstream name;
    name << counter++ << ".ogg";
    auto outputName = name.str();

    auto fn = [](piper::PiperConfig *config, piper::Voice *voice, uint32_t id, string inputText, string outputName) {
      piper::SynthesisResult result;
      vector<int16_t> audioBuffer;
      textToAudio(*config, *voice, inputText, audioBuffer, result, NULL);
      spdlog::info("Real-time factor: {} (infer={} sec, audio={} sec)",
                  result.realTimeFactor, result.inferSeconds,
                  result.audioSeconds);
      auto output = convertWavToOgg(*voice, audioBuffer);

      // TODO: if/when we multi-thread this, we'll need a lock around writing to
      // the output.
      if (!_isatty(_fileno(stdout))) {
        // Write the following data:
        //   output_size: 4 bytes
        //   id:          4 bytes
        //   output:      `output_size` bytes

        string outputStr = output.str();
        uint32_t outputSize = (uint32_t)outputStr.size();

        // Write size of the ogg binary data.
        fwrite((const char*)&outputSize, sizeof(outputSize), 1, stdout);
        spdlog::info("Wrote output size {}", outputSize);
        // Write the "id" member from above back to the output, with a size-prefix.
        fwrite((const char*)&id, sizeof(id), 1, stdout);
        spdlog::info("Wrote id {}", id);
        // Write the output ogg as a binary blob.
        fwrite(outputStr.data(), 1, outputSize, stdout);
        fflush(stdout);
        spdlog::info("Wrote output of size {}", outputSize);
      } else {
        // Output audio to OGG file
        ofstream audioFile(outputName, ios::binary);
        audioFile << output.rdbuf();
      }
    };

    // TODO: espeak-ng uses global variables so is not thread-safe. We can
    // work around it by merging with a fork that supports a context (see
    // https://github.com/espeak-ng/espeak-ng/issues/1527) or simpler, we can
    // do all of the phonemization on the main thread, since the
    // phoneme->audio conversion by onnx is thread-safe.
    if (false) {
      thread_pool->enqueue(move(fn), &piperConfig, voice, id, inputText, outputName);
    } else {
      fn(&piperConfig, voice, id, inputText, outputName);
    }
  }

  spdlog::info("waiting for thread pool to finish.");
  thread_pool.reset();

  spdlog::info("piper_ogg done.");
  piper::terminate(piperConfig);
  return 0;
}

/// Copied from vorbis encoding example
stringstream convertWavToOgg(piper::Voice &voice, const vector<int16_t> &input) {
  size_t inputOffset = 0;
  stringstream output;

  ogg_stream_state os; /* take physical pages, weld into a logical stream of packets */
  ogg_page         og; /* one Ogg bitstream page.  Vorbis packets are inside */
  ogg_packet       op; /* one raw packet of data for decode */
  vorbis_info      vi; /* struct that stores all the static vorbis bitstream settings */
  vorbis_comment   vc; /* struct that stores all the user comments */
  vorbis_dsp_state vd; /* central working state for the packet->PCM decoder */
  vorbis_block     vb; /* local working space for packet->PCM decode */

  int eos=0, ret;

  auto startTime = chrono::steady_clock::now();
  vorbis_info_init(&vi);
  ret = vorbis_encode_init_vbr(&vi, voice.synthesisConfig.channels, voice.synthesisConfig.sampleRate, .4f);
  if (ret != 0) {
    spdlog::error("vorbis_encode_init_vbr failed with {}", ret);
    return {};
  }

  vorbis_comment_init(&vc);
  vorbis_comment_add_tag(&vc, "ENCODER", "piper_ogg.exe");

  /* set up the analysis state and auxiliary encoding storage */
  vorbis_analysis_init(&vd, &vi);
  vorbis_block_init(&vd, &vb);

  /* set up our packet->stream encoder */
  ogg_stream_init(&os, 0xFEEDBA6);

  /* Vorbis streams begin with three headers; the initial header (with
     most of the codec setup parameters) which is mandated by the Ogg
     bitstream spec.  The second header holds any comment fields.  The
     third header holds the bitstream codebook.  We merely need to
     make the headers, then pass them to libvorbis one at a time;
     libvorbis handles the additional Ogg bitstream constraints */

  {
    ogg_packet header;
    ogg_packet header_comm;
    ogg_packet header_code;

    vorbis_analysis_headerout(&vd, &vc, &header, &header_comm, &header_code);
    ogg_stream_packetin(&os, &header); /* automatically placed in its own page */
    ogg_stream_packetin(&os, &header_comm);
    ogg_stream_packetin(&os, &header_code);

    /* This ensures the actual
     * audio data will start on a new page, as per spec
     */
    while (!eos) {
      int result = ogg_stream_flush(&os, &og);
      if (result == 0) break;
      output.write((char*)og.header, og.header_len);
      output.write((char*)og.body, og.body_len);
    }
  }

  const size_t sampleBufferSize = 1024;
  int channels = voice.synthesisConfig.channels;
  while (!eos) {
    size_t inputEnd = min(inputOffset + sampleBufferSize, input.size());
    size_t samples = inputEnd - inputOffset;
    const int16_t *readbuffer = input.data() + inputOffset;
    inputOffset += samples;

    if (samples == 0) {
      /* end of file.  this can be done implicitly in the mainline,
         but it's easier to see here in non-clever fashion.
         Tell the library we're at end of stream so that it can handle
         the last frame and mark end of stream in the output properly */
      vorbis_analysis_wrote(&vd, 0);
    } else {
      /* data to encode */
      /* expose the buffer to submit data */
      float **buffer = vorbis_analysis_buffer(&vd, sampleBufferSize);

      /* uninterleave samples */
      long i;
      for (i = 0; i < samples / channels; i++) {
        for (int ch = 0; ch < channels; ++ch) {
          buffer[ch][i] = readbuffer[i * channels + ch] / 32768.f;
        }
      }

      /* tell the library how much we actually submitted */
      vorbis_analysis_wrote(&vd, i);
    }

    /* vorbis does some data preanalysis, then divvies up blocks for
       more involved (potentially parallel) processing.  Get a single
       block for encoding now */
    while (vorbis_analysis_blockout(&vd, &vb) == 1) {
      /* analysis, assume we want to use bitrate management */
      vorbis_analysis(&vb, NULL);
      vorbis_bitrate_addblock(&vb);

      while (vorbis_bitrate_flushpacket(&vd, &op)) {
        /* weld the packet into the bitstream */
        ogg_stream_packetin(&os, &op);

        /* write out pages (if any) */
        while (!eos) {
          int result = ogg_stream_pageout(&os, &og);
          if (result == 0) break;
          output.write((char*)og.header, og.header_len);
          output.write((char*)og.body, og.body_len);

          /* this could be set above, but for illustrative purposes, I do
             it here (to show that vorbis does know where the stream ends) */
          if (ogg_page_eos(&og)) eos = 1;
        }
      }
    }
  }

  /* clean up and exit.  vorbis_info_clear() must be called last */
  ogg_stream_clear(&os);
  vorbis_block_clear(&vb);
  vorbis_dsp_clear(&vd);
  vorbis_comment_clear(&vc);
  vorbis_info_clear(&vi);

  auto endTime = chrono::steady_clock::now();
  spdlog::info("Converted to OGG in {} second(s)", chrono::duration<double>(endTime - startTime).count());

  return output;
}
