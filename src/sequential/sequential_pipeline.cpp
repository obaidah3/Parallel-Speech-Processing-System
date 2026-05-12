#include "audio.h"
#include "dsp.h"
#include "fft_utils.h"
#include "features.h"
#include "timing.h"
#include <vector>
#include <complex>
#include <iostream>

// Sequential audio processing pipeline
void runSequentialPipeline(const std::string& inputFile, const std::string& outputFile, const std::string& featuresFile) {
    Timer timer;
    timer.start();

    // Read audio
    auto [samples, sampleRate] = readWavMono(inputFile);
    std::cout << "Sequential: Read " << samples.size() << " samples at " << sampleRate << " Hz" << std::endl;

    // Preprocessing
    normalizeAudio(samples);
    removeDcOffset(samples);
    applyPreEmphasis(samples);

    // STFT
    std::vector<std::vector<std::complex<float>>> spectrogram;
    computeSTFT(samples, spectrogram);

    // Filtering
    for (auto& frame : spectrogram) {
        applySpeechFilter(frame, sampleRate);
    }

    // Feature extraction
    extractFeatures(spectrogram, sampleRate, featuresFile);

    // ISTFT
    std::vector<float> reconstructed;
    computeISTFT(spectrogram, reconstructed);

    // Write output
    writeWavMono(outputFile, reconstructed, sampleRate);

    timer.stop();
    std::cout << "Sequential processing time: " << timer.elapsedSeconds() << " seconds" << std::endl;
}