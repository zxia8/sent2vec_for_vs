/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <iostream>

#include "fasttext.h"
#include "args.h"

using namespace fasttext;

void printUsage() {
    std::cout
        << "usage: fasttext <command> <args>\n\n"
        << "The commands supported by fasttext are:\n\n"
        << "  supervised          train a supervised classifier\n"
        << "  sent2vec            train unsupervised sentence embeddings\n"
        << "  test                evaluate a supervised classifier\n"
        << "  predict             predict most likely labels\n"
        << "  predict-prob        predict most likely labels with probabilities\n"
        << "  skipgram            train a skipgram model\n"
        << "  cbow                train a cbow model\n"
        << "  print-vectors       print vectors given a trained model\n"
        << std::endl;
}

void printTestUsage() {
    std::cout
        << "usage: fasttext test <model> <test-data> [<k>]\n\n"
        << "  <model>      model filename\n"
        << "  <test-data>  test data filename (if -, read from stdin)\n"
        << "  <k>          (optional; 1 by default) predict top k labels\n"
        << std::endl;
}

void printPredictUsage() {
    std::cout
        << "usage: fasttext predict[-prob] <model> <test-data> [<k>]\n\n"
        << "  <model>      model filename\n"
        << "  <test-data>  test data filename (if -, read from stdin)\n"
        << "  <k>          (optional; 1 by default) predict top k labels\n"
        << std::endl;
}

void printPrintVectorsUsage() {
    std::cout
        << "usage: fasttext print-vectors <model>\n\n"
        << "  <model>      model filename\n"
        << std::endl;
}

void printPrintNgramsUsage() {
    std::cout
        << "usage: fasttext print-ngrams <model> <word>\n\n"
        << "  <model>      model filename\n"
        << "  <word>       word to print\n"
        << std::endl;
}

void test(int argc, char** argv) {
    int32_t k;
    if (argc == 4) {
        k = 1;
    }
    else if (argc == 5) {
        k = atoi(argv[4]);
    }
    else {
        printTestUsage();
        exit(EXIT_FAILURE);
    }
    FastText fasttext;
    fasttext.loadModel(std::string(argv[2]));
    std::string infile(argv[3]);
    if (infile == "-") {
        fasttext.test(std::cin, k);
    }
    else {
        std::ifstream ifs(infile);
        if (!ifs.is_open()) {
            std::cerr << "Test file cannot be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }
        fasttext.test(ifs, k);
        ifs.close();
    }
    exit(0);
}

void predict(int argc, char** argv) {
    int32_t k;
    if (argc == 4) {
        k = 1;
    }
    else if (argc == 5) {
        k = atoi(argv[4]);
    }
    else {
        printPredictUsage();
        exit(EXIT_FAILURE);
    }
    bool print_prob = std::string(argv[1]) == "predict-prob";
    FastText fasttext;
    fasttext.loadModel(std::string(argv[2]));

    std::string infile(argv[3]);
    if (infile == "-") {
        fasttext.predict(std::cin, k, print_prob);
    }
    else {
        std::ifstream ifs(infile);
        if (!ifs.is_open()) {
            std::cerr << "Input file cannot be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }
        fasttext.predict(ifs, k, print_prob);
        ifs.close();
    }

    exit(0);
}

void printVectors(int argc, char** argv) {
    if (argc != 3) {
        printPrintVectorsUsage();
        exit(EXIT_FAILURE);
    }
    FastText fasttext;
    fasttext.loadModel(std::string(argv[2]));
    fasttext.printVectors();
    exit(0);
}

void printNgrams(int argc, char** argv) {
    if (argc != 4) {
        printPrintNgramsUsage();
        exit(EXIT_FAILURE);
    }
    FastText fasttext;
    fasttext.loadModel(std::string(argv[2]));
    fasttext.ngramVectors(std::string(argv[3]));
    exit(0);
}

void train(int argc, char** argv) {
    std::shared_ptr<Args> a = std::make_shared<Args>();
    a->parseArgs(argc, argv);
    FastText fasttext;
    fasttext.train(a);
}

void test_nn(int argc, char** argv) {
    FastText fasttext;
    fasttext.loadModel(std::string(argv[2]));
    fasttext.test_nn();
    exit(0);
}

void p(int argc, char** argv) {
    FastText fasttext;
    fasttext.loadModel(std::string(argv[2]));
    fasttext.print_sentence_vectors(argv[3], argv[4]);
    exit(0);
}

int main(int argc, char** argv) {
    //std::string mdl = argv[1];
    //FastText fasttext;
    //fasttext.loadModel(mdl);
    //fasttext.saveVectors();
    //fasttext.saveDict();
    if (argc < 1) {
        printUsage();
        exit(EXIT_FAILURE);
    }
    std::string command(argv[1]);
    if (command == "skipgram" || command == "cbow" || command == "supervised" ||
        command == "sent2vec") {
        train(argc, argv);
    }
    else if (command == "test") {
        test(argc, argv);
    }
    else if (command == "print-vectors") {
        printVectors(argc, argv);
    }
    else if (command == "print-ngrams") {
        printNgrams(argc, argv);
    }
    else if (command == "predict" || command == "predict-prob") {
        predict(argc, argv);
    }
    else if (command == "test-nn") {
        test_nn(argc, argv);
    }
    else if (command == "generate_s2v") {
        p(argc, argv);
    }
    else {
        printUsage();
        exit(EXIT_FAILURE);
    }
    return 0;
}
