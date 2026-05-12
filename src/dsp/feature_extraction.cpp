// Create mel filterbank
std::vector<std::vector<float>> createMelFilterbank(int numFilters, int fftSize, int sampleRate) {
    std::vector<std::vector<float>> filterbank(numFilters, std::vector<float>(fftSize / 2 + 1, 0.0f));

    float fMin = 0.0f;
    float fMax = sampleRate / 2.0f;

    // Convert Hz to mel
    auto hzToMel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
    auto melToHz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };

    float melMin = hzToMel(fMin);
    float melMax = hzToMel(fMax);

    std::vector<float> melPoints(numFilters + 2);
    for (int i = 0; i < numFilters + 2; ++i) {
        melPoints[i] = melMin + (melMax - melMin) * i / (numFilters + 1);
    }

    std::vector<float> hzPoints(numFilters + 2);
    for (int i = 0; i < numFilters + 2; ++i) {
        hzPoints[i] = melToHz(melPoints[i]);
    }

    std::vector<int> binPoints(numFilters + 2);
    for (int i = 0; i < numFilters + 2; ++i) {
        binPoints[i] = std::floor((fftSize + 1) * hzPoints[i] / sampleRate);
    }

    for (int m = 1; m <= numFilters; ++m) {
        int fLeft = binPoints[m - 1];
        int fCenter = binPoints[m];
        int fRight = binPoints[m + 1];

        for (int k = fLeft; k < fCenter; ++k) {
            if (k >= 0 && k < static_cast<int>(filterbank[m - 1].size())) {
                filterbank[m - 1][k] = (k - fLeft) / static_cast<float>(fCenter - fLeft);
            }
        }

        for (int k = fCenter; k < fRight; ++k) {
            if (k >= 0 && k < static_cast<int>(filterbank[m - 1].size())) {
                filterbank[m - 1][k] = (fRight - k) / static_cast<float>(fRight - fCenter);
            }
        }
    }

    return filterbank;
}

// Calculate RMS energy
float calculateRMS(const std::vector<float>& samples) {
    float sum = 0.0f;
    for (float sample : samples) {
        sum += sample * sample;
    }
    return std::sqrt(sum / samples.size());
}

// Calculate Zero Crossing Rate
float calculateZCR(const std::vector<float>& samples) {
    int crossings = 0;
    for (size_t i = 1; i < samples.size(); ++i) {
        if ((samples[i-1] >= 0 && samples[i] < 0) || (samples[i-1] < 0 && samples[i] >= 0)) {
            crossings++;
        }
    }
    return static_cast<float>(crossings) / samples.size();
}

// Calculate spectral centroid
float calculateSpectralCentroid(const std::vector<std::complex<float>>& frame, int sampleRate) {
    float numerator = 0.0f;
    float denominator = 0.0f;
    int fftSize = (frame.size() - 1) * 2;

    for (size_t k = 0; k < frame.size(); ++k) {
        float magnitude = std::abs(frame[k]);
        float freq = k * sampleRate / static_cast<float>(fftSize);
        numerator += freq * magnitude;
        denominator += magnitude;
    }

    return denominator > 0 ? numerator / denominator : 0.0f;
}

// Calculate spectral bandwidth
float calculateSpectralBandwidth(const std::vector<std::complex<float>>& frame, int sampleRate, float centroid) {
    float numerator = 0.0f;
    float denominator = 0.0f;
    int fftSize = (frame.size() - 1) * 2;

    for (size_t k = 0; k < frame.size(); ++k) {
        float magnitude = std::abs(frame[k]);
        float freq = k * sampleRate / static_cast<float>(fftSize);
        numerator += (freq - centroid) * (freq - centroid) * magnitude;
        denominator += magnitude;
    }

    return denominator > 0 ? std::sqrt(numerator / denominator) : 0.0f;
}

// Calculate spectral flatness
float calculateSpectralFlatness(const std::vector<std::complex<float>>& frame) {
    float geometricMean = 1.0f;
    float arithmeticMean = 0.0f;
    int count = 0;

    for (const auto& bin : frame) {
        float magnitude = std::abs(bin);
        if (magnitude > 0) {
            geometricMean *= magnitude;
            arithmeticMean += magnitude;
            count++;
        }
    }

    if (count == 0) return 0.0f;

    geometricMean = std::pow(geometricMean, 1.0f / count);
    arithmeticMean /= count;

    return arithmeticMean > 0 ? geometricMean / arithmeticMean : 0.0f;
}

// Extract MFCC coefficients using mel filterbank and DCT
std::vector<float> extractMFCC(const std::vector<std::complex<float>>& frame, int sampleRate, int numCoeffs) {
    int fftSize = (frame.size() - 1) * 2;
    int numFilters = 26; // Standard number of mel filters

    // Convert to power spectrum
    std::vector<float> powerSpec(frame.size());
    for (size_t i = 0; i < frame.size(); ++i) {
        powerSpec[i] = std::norm(frame[i]); // |z|^2
    }

    // Mel filterbank
    std::vector<std::vector<float>> melFilterbank = createMelFilterbank(numFilters, fftSize, sampleRate);
    std::vector<float> melEnergies(numFilters, 0.0f);

    for (int f = 0; f < numFilters; ++f) {
        for (size_t k = 0; k < powerSpec.size(); ++k) {
            melEnergies[f] += powerSpec[k] * melFilterbank[f][k];
        }
        // Log compression
        melEnergies[f] = std::log(std::max(melEnergies[f], 1e-10f));
    }

    // Discrete Cosine Transform (DCT)
    std::vector<float> mfcc(numCoeffs, 0.0f);
    for (int n = 0; n < numCoeffs; ++n) {
        for (int k = 0; k < numFilters; ++k) {
            mfcc[n] += melEnergies[k] * std::cos(M_PI * n * (k + 0.5f) / numFilters);
        }
    }

    return mfcc;
}

// Extract features from spectrogram and save to CSV
void extractFeatures(const Spectrogram& spectrogram, int sampleRate, const std::string& outputFile) {
    std::ofstream file(outputFile);
    file << "Frame,RMS,ZCR,Centroid,Bandwidth,Flatness";

    for (int i = 0; i < 13; ++i) {
        file << ",MFCC" << i;
    }
    file << "\n";

    int numFrames = spectrogram.numFrames();
    std::vector<std::string> frameLines(numFrames);

    #pragma omp parallel for
    for (int frame = 0; frame < numFrames; ++frame) {
        // Extract frame data
        std::vector<std::complex<float>> specFrame(spectrogram.numBins());
        for (int bin = 0; bin < spectrogram.numBins(); ++bin) {
            specFrame[bin] = spectrogram(frame, bin);
        }

        // Convert complex to real for RMS and ZCR (simplified)
        std::vector<float> realFrame(specFrame.size());
        for (size_t i = 0; i < specFrame.size(); ++i) {
            realFrame[i] = specFrame[i].real();
        }

        float rms = calculateRMS(realFrame);
        float zcr = calculateZCR(realFrame);
        float centroid = calculateSpectralCentroid(specFrame, sampleRate);
        float bandwidth = calculateSpectralBandwidth(specFrame, sampleRate, centroid);
        float flatness = calculateSpectralFlatness(specFrame);
        auto mfcc = extractMFCC(specFrame, sampleRate, 13);

        std::ostringstream oss;
        oss << frame << "," << rms << "," << zcr << "," << centroid << "," << bandwidth << "," << flatness;
        for (float coeff : mfcc) {
            oss << "," << coeff;
        }
        frameLines[frame] = oss.str();
    }

    // Write results sequentially to avoid I/O contention
    for (const auto& line : frameLines) {
        file << line << "\n";
    }

    file.close();
}