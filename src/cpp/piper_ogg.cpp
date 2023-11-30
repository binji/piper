#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"
#include "piper.hpp"
#include "minivorbis.h"

using namespace std;

int main(int argc, char *argv[]) {
  piper::PiperConfig piperConfig;
  piper::Voice voice;

  if (argc < 2) {
    std::cerr << "Need voice model path" << std::endl;
    return 1;
  }

  if (argc < 3) {
    std::cerr << "Need espeak-ng-data path" << std::endl;
    return 1;
  }

  if (argc < 4) {
    std::cerr << "Need output WAV path" << std::endl;
    return 1;
  }

  if (argc < 5) {
    std::cerr << "Need input text" << std::endl;
    return 1;
  }

  auto modelPath = std::string(argv[1]);
  piperConfig.eSpeakDataPath = std::string(argv[2]);
  auto outputPath = std::string(argv[3]);
  auto inputText = std::string(argv[4]);

  optional<piper::SpeakerId> speakerId;
  loadVoice(piperConfig, modelPath, modelPath + ".json", voice, speakerId,
            false);
  piper::initialize(piperConfig);

  if (voice.synthesisConfig.sampleWidth != 2) {
    std::cerr << "ERROR: expected sample width to be 2" << std::endl;
    return EXIT_FAILURE;
  }

  piper::SynthesisResult result;
  std::vector<int16_t> audioBuffer;
  textToAudio(piperConfig, voice, inputText, audioBuffer, result, NULL);
  piper::terminate(piperConfig);

  const auto &input = audioBuffer;
  size_t inputOffset = 0;
  std::stringstream output;

  /// Copied from vorbis encoding example

  ogg_stream_state os; /* take physical pages, weld into a logical stream of packets */
  ogg_page         og; /* one Ogg bitstream page.  Vorbis packets are inside */
  ogg_packet       op; /* one raw packet of data for decode */
  vorbis_info      vi; /* struct that stores all the static vorbis bitstream settings */
  vorbis_comment   vc; /* struct that stores all the user comments */
  vorbis_dsp_state vd; /* central working state for the packet->PCM decoder */
  vorbis_block     vb; /* local working space for packet->PCM decode */

  int eos=0, ret;

  vorbis_info_init(&vi);
  ret = vorbis_encode_init_vbr(&vi, voice.synthesisConfig.channels, voice.synthesisConfig.sampleRate, .4f);
  if (ret != 0) {
    std::cerr << "ERROR: vorbis_encode_init_vbr failed with " << ret << std::endl;
    return EXIT_FAILURE;
  }

  vorbis_comment_init(&vc);
  vorbis_comment_add_tag(&vc, "ENCODER", "piper_ogg.exe");

  /* set up the analysis state and auxiliary encoding storage */
  vorbis_analysis_init(&vd, &vi);
  vorbis_block_init(&vd, &vb);

  /* set up our packet->stream encoder */
  /* pick a random serial number; that way we can more likely build
     chained streams just by concatenation */
  srand(time(NULL));
  ogg_stream_init(&os, rand());

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
    size_t inputEnd = std::min(inputOffset + sampleBufferSize, input.size());
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

  // Output audio to OGG file
  ofstream audioFile(outputPath, ios::binary);
  audioFile << output.rdbuf();

  std::cout << "OK" << std::endl;

  return EXIT_SUCCESS;
}
