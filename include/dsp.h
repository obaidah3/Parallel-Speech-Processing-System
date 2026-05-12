#ifndef DSP_H
#define DSP_H

#include <vector>

// DSP preprocessing functions
void normalizeAudio(std::vector<float>& samples);
void removeDcOffset(std::vector<float>& samples);
void applyPreEmphasis(std::vector<float>& samples, float alpha = 0.97f);

#endif