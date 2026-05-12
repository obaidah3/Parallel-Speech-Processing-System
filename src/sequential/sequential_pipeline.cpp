#include "audio.h"
#include "dsp.h"
#include "fft_utils.h"
#include "audio_features.h"
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
    Spectrogram spectrogram(0, 0);
    computeSTFT(samples, spectrogram);

    // Filtering
    for (int frame = 0; frame < spectrogram.numFrames(); ++frame) {
        std::vector<std::complex<float>> frameData(spectrogram.numBins());
        for (int bin = 0; bin < spectrogram.numBins(); ++bin) {
            frameData[bin] = spectrogram(frame, bin);
        }
        applySpeechFilter(frameData, sampleRate);
        for (int bin = 0; bin < spectrogram.numBins(); ++bin) {
            spectrogram(frame, bin) = frameData[bin];
        }
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