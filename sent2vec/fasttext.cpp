/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "fasttext.h"
#include "hnswlib/hnswlib.h"

#include <math.h>
#include <iostream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

namespace fasttext {

    void FastText::getVector(Vector& vec, const std::string& word) {
        const std::vector<int32_t>& ngrams = dict_->getNgrams(word);
        vec.zero();
        for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
            vec.addRow(*input_, *it);
        }
        if (ngrams.size() > 0) {
            vec.mul(1.0 / ngrams.size());
        }
    }

    void FastText::saveVectors() {
        std::ofstream ofs(args_->output + ".vec");
        //std::ofstream ofs("D:\\s2v_vectors.vec");
        if (!ofs.is_open()) {
            std::cout << "Error opening file for saving vectors." << std::endl;
            exit(EXIT_FAILURE);
        }
        ofs << dict_->nwords() << " " << args_->dim << std::endl;
        Vector vec(args_->dim);
        for (int32_t i = 0; i < dict_->nwords(); i++) {
            std::string word = dict_->getWord(i);
            getVector(vec, word);
            ofs << word << " " << vec << std::endl;
        }
        ofs.close();
    }

    void FastText::saveOutput() {
        std::ofstream ofs(args_->output + ".output");
        if (!ofs.is_open()) {
            std::cout << "Error opening file for saving vectors." << std::endl;
            exit(EXIT_FAILURE);
        }
        ofs << dict_->nwords() << " " << args_->dim << std::endl;
        Vector vec(args_->dim);
        for (int32_t i = 0; i < dict_->nwords(); i++) {
            std::string word = dict_->getWord(i);
            vec.zero();
            vec.addRow(*output_, i);
            ofs << word << " " << vec << std::endl;
        }
        ofs.close();
    }

    void FastText::saveDict() {
        //std::ofstream ofs(args_->output + ".txt");
        std::ofstream ofs("D:\\s2v_term_frequency.txt");
        if (!ofs.is_open()) {
            std::cout << "Error opening file for saving tf." << std::endl;
            exit(EXIT_FAILURE);
        }
        for (int32_t i = 0; i < dict_->nwords(); i++) {
            std::string word = dict_->getWord(i);
            int64_t frequency = dict_->getTokenCount(i);

            ofs << word << " " << frequency << "\n";
        }
        ofs.close();
    }

    void FastText::saveModel() {
        std::ofstream ofs(args_->output + ".bin", std::ofstream::binary);
        if (!ofs.is_open()) {
            std::cerr << "Model file cannot be opened for saving!" << std::endl;
            exit(EXIT_FAILURE);
        }
        args_->save(ofs);
        dict_->save(ofs);
        input_->save(ofs);
        output_->save(ofs);
        ofs.close();
    }

    void FastText::loadModel(const std::string& filename) {
        std::ifstream ifs(filename, std::ifstream::binary);
        if (!ifs.is_open()) {
            std::cerr << "Model file cannot be opened for loading!" << std::endl;
            exit(EXIT_FAILURE);
        }
        loadModel(ifs);
        ifs.close();
    }

    void FastText::loadModel(std::istream& in) {
        args_ = std::make_shared<Args>();
        dict_ = std::make_shared<Dictionary>(args_);
        input_ = std::make_shared<Matrix>();
        output_ = std::make_shared<Matrix>();
        args_->load(in);
        dict_->load(in);
        input_->load(in);
        output_->load(in);
        model_ = std::make_shared<Model>(input_, output_, args_, 0);
        if (args_->model == model_name::sup) {
            model_->setTargetCounts(dict_->getCounts(entry_type::label));
        }
        else {
            model_->setTargetCounts(dict_->getCounts(entry_type::word));
        }
    }

    void FastText::printInfo(real progress, real loss) {
        real t = real(clock() - start) / CLOCKS_PER_SEC;
        real wst = real(tokenCount) / t;
        real lr = args_->lr * (1.0 - progress);
        int eta = int(t / progress * (1 - progress) / args_->thread);
        int etah = eta / 3600;
        int etam = (eta - etah * 3600) / 60;
        std::cout << std::fixed;
        std::cout << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
        std::cout << "  words/sec/thread: " << std::setprecision(0) << wst;
        std::cout << "  lr: " << std::setprecision(6) << lr;
        std::cout << "  loss: " << std::setprecision(6) << loss;
        std::cout << "  eta: " << etah << "h" << etam << "m ";
        std::cout << std::flush;
    }

    void FastText::supervised(Model& model, real lr,
        const std::vector<int32_t>& line,
        const std::vector<int32_t>& labels) {
        if (labels.size() == 0 || line.size() == 0) return;
        std::uniform_int_distribution<> uniform(0, labels.size() - 1);
        int32_t i = uniform(model.rng);
        model.update(line, labels[i], lr);
    }

    void FastText::cbow(Model& model, real lr,
        const std::vector<int32_t>& line) {
        std::vector<int32_t> bow;
        std::uniform_int_distribution<> uniform(1, args_->ws);
        for (int32_t w = 0; w < line.size(); w++) {
            int32_t boundary = uniform(model.rng);
            bow.clear();
            for (int32_t c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    const std::vector<int32_t>& ngrams = dict_->getNgrams(line[w + c]);
                    bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
                }
            }
            model.update(bow, line[w], lr);
        }
    }

    void FastText::skipgram(Model& model, real lr,
        const std::vector<int32_t>& line) {
        std::uniform_int_distribution<> uniform(1, args_->ws);
        for (int32_t w = 0; w < line.size(); w++) {
            int32_t boundary = uniform(model.rng);
            const std::vector<int32_t>& ngrams = dict_->getNgrams(line[w]);
            for (int32_t c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    model.update(ngrams, line[w + c], lr);
                }
            }
        }
    }

    void FastText::sent2vec(Model& model, real lr, const std::vector<int32_t>& line) {
        if (line.size() <= 1) return;
        std::vector<int32_t> context;
        std::uniform_real_distribution<> uniform(0, 1);
        for (int32_t i = 0; i < line.size(); ++i) {
            if (uniform(model.rng) > dict_->getPDiscard(line[i]) || dict_->getTokenCount(line[i]) < args_->minCountLabel)
                continue;
            context = line;
            context[i] = 0;
            dict_->addNgrams(context, args_->wordNgrams, args_->dropoutK, model.rng);
            model.update(context, line[i], lr);
        }
    }

    void FastText::test(std::istream& in, int32_t k) {
        int32_t nexamples = 0, nlabels = 0;
        double precision = 0.0;
        std::vector<int32_t> line, labels;

        while (in.peek() != EOF) {
            dict_->getLine(in, line, labels, model_->rng);
            dict_->addNgrams(line, args_->wordNgrams);
            if (labels.size() > 0 && line.size() > 0) {
                std::vector<std::pair<real, int32_t>> modelPredictions;
                model_->predict(line, k, modelPredictions);
                for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
                    if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
                        precision += 1.0;
                    }
                }
                nexamples++;
                nlabels += labels.size();
            }
        }
        std::cout << std::setprecision(3);
        std::cout << "P@" << k << ": " << precision / (k * nexamples) << std::endl;
        std::cout << "R@" << k << ": " << precision / nlabels << std::endl;
        std::cout << "Number of examples: " << nexamples << std::endl;
    }

    void FastText::predict(std::istream& in, int32_t k,
        std::vector<std::pair<real, std::string>>& predictions) const {
        std::vector<int32_t> words, labels;
        dict_->getLine(in, words, labels, model_->rng);
        dict_->addNgrams(words, args_->wordNgrams);
        if (words.empty()) return;
        Vector hidden(args_->dim);
        Vector output(dict_->nlabels());
        std::vector<std::pair<real, int32_t>> modelPredictions;
        model_->predict(words, k, modelPredictions, hidden, output);
        predictions.clear();
        for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
            predictions.push_back(std::make_pair(it->first, dict_->getLabel(it->second)));
        }
    }

    void FastText::predict(std::istream& in, int32_t k, bool print_prob) {
        std::vector<std::pair<real, std::string>> predictions;
        while (in.peek() != EOF) {
            predict(in, k, predictions);
            if (predictions.empty()) {
                std::cout << "n/a" << std::endl;
                continue;
            }
            for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
                if (it != predictions.cbegin()) {
                    std::cout << ' ';
                }
                std::cout << it->second;
                if (print_prob) {
                    std::cout << ' ' << exp(it->first);
                }
            }
            std::cout << std::endl;
        }
    }

    void FastText::wordVectors() {
        std::string word;
        Vector vec(args_->dim);
        while (std::cin >> word) {
            getVector(vec, word);
            std::cout << word << " " << vec << std::endl;
        }
    }

    void FastText::ngramVectors(std::string word) {
        std::vector<int32_t> ngrams;
        std::vector<std::string> substrings;
        Vector vec(args_->dim);
        dict_->getNgrams(word, ngrams, substrings);
        for (int32_t i = 0; i < ngrams.size(); i++) {
            vec.zero();
            if (ngrams[i] >= 0) {
                vec.addRow(*input_, ngrams[i]);
            }
            std::cout << substrings[i] << " " << vec << std::endl;
        }
    }

    void FastText::textVectors() {
        std::vector<int32_t> line, labels;
        Vector vec(args_->dim);
        while (std::cin.peek() != EOF) {
            dict_->getLine(std::cin, line, labels, model_->rng);
            dict_->addNgrams(line, args_->wordNgrams);
            vec.zero();
            for (auto it = line.cbegin(); it != line.cend(); ++it) {
                vec.addRow(*input_, *it);
            }
            if (!line.empty()) {
                vec.mul(1.0 / line.size());
            }
            std::cout << vec << std::endl;
        }
    }

    void FastText::printVectors() {
        if (args_->model == model_name::sup || args_->model == model_name::sent2vec) {
            textVectors();
        }
        else {
            wordVectors();
        }
    }

    void FastText::trainThread(int32_t threadId) {
        std::ifstream ifs(args_->input);
        utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

        Model model(input_, output_, args_, threadId);
        if (args_->model == model_name::sup) {
            model.setTargetCounts(dict_->getCounts(entry_type::label));
        }
        else {
            model.setTargetCounts(dict_->getCounts(entry_type::word));
        }

        const int64_t ntokens = dict_->ntokens();
        int64_t localTokenCount = 0;
        std::vector<int32_t> line, labels;
        while (tokenCount < args_->epoch * ntokens) {
            real progress = real(tokenCount) / (args_->epoch * ntokens);
            real lr = args_->lr * (1.0 - progress);
            localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
            if (args_->model == model_name::sup) {
                dict_->addNgrams(line, args_->wordNgrams);
                supervised(model, lr, line, labels);
            }
            else if (args_->model == model_name::sent2vec) {
                sent2vec(model, lr, line);
            }
            else if (args_->model == model_name::cbow) {
                cbow(model, lr, line);
            }
            else if (args_->model == model_name::sg) {
                skipgram(model, lr, line);
            }
            if (localTokenCount > args_->lrUpdateRate) {
                tokenCount += localTokenCount;
                localTokenCount = 0;
                if (threadId == 0 && args_->verbose > 1) {
                    printInfo(progress, model.getLoss());
                }
            }
        }
        if (threadId == 0 && args_->verbose > 0) {
            printInfo(1.0, model.getLoss());
            std::cout << std::endl;
        }
        ifs.close();
    }

    void FastText::loadVectors(std::string filename) {
        std::ifstream in(filename);
        std::vector<std::string> words;
        std::shared_ptr<Matrix> mat; // temp. matrix for pretrained vectors
        int64_t n, dim;
        if (!in.is_open()) {
            std::cerr << "Pretrained vectors file cannot be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }
        in >> n >> dim;
        if (dim != args_->dim) {
            std::cerr << "Dimension of pretrained vectors does not match -dim option"
                << std::endl;
            exit(EXIT_FAILURE);
        }
        mat = std::make_shared<Matrix>(n, dim);
        for (size_t i = 0; i < n; i++) {
            std::string word;
            in >> word;
            words.push_back(word);
            dict_->add(word);
            for (size_t j = 0; j < dim; j++) {
                in >> mat->data_[i * dim + j];
            }
        }
        in.close();

        dict_->threshold(1, 0);
        input_ = std::make_shared<Matrix>(dict_->nwords() + args_->bucket, args_->dim);
        input_->uniform(1.0 / args_->dim);

        for (size_t i = 0; i < n; i++) {
            int32_t idx = dict_->getId(words[i]);
            if (idx < 0 || idx >= dict_->nwords()) continue;
            for (size_t j = 0; j < dim; j++) {
                input_->data_[idx * dim + j] = mat->data_[i * dim + j];
            }
        }
    }

    void FastText::train(std::shared_ptr<Args> args) {
        args_ = args;
        dict_ = std::make_shared<Dictionary>(args_);
        if (args_->input == "-") {
            // manage expectations
            std::cerr << "Cannot use stdin for training!" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::ifstream ifs(args_->input);
        if (!ifs.is_open()) {
            std::cerr << "Input file cannot be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }
        dict_->readFromFile(ifs);
        ifs.close();

        if (args_->pretrainedVectors.size() != 0) {
            loadVectors(args_->pretrainedVectors);
        }
        else {
            input_ = std::make_shared<Matrix>(dict_->nwords() + args_->bucket, args_->dim);
            input_->uniform(1.0 / args_->dim);
        }

        if (args_->model == model_name::sup) {
            output_ = std::make_shared<Matrix>(dict_->nlabels(), args_->dim);
        }
        else {
            output_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);
        }
        output_->zero();

        start = clock();
        tokenCount = 0;
        std::vector<std::thread> threads;
        for (int32_t i = 0; i < args_->thread; i++) {
            threads.push_back(std::thread([=]() { trainThread(i); }));
        }
        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
        model_ = std::make_shared<Model>(input_, output_, args_, 0);

        saveModel();
        if (args_->model != model_name::sup && args_->model != model_name::sent2vec) {
            saveVectors();
            if (args_->saveOutput > 0) {
                saveOutput();
            }
        }
    }
    void FastText::find_knn_by_query(const std::string& content, hnswlib::HierarchicalNSW<float>* appr_alg, int k, std::vector<std::string> sents) {
        std::vector<int32_t> line, labels;
        Vector vec(args_->dim);
        std::istringstream ss(content);
        dict_->getLine(ss, line, labels, model_->rng);
        dict_->addNgrams(line, args_->wordNgrams);
        vec.zero();
        for (int it : line) {
            vec.addRow(*input_, it);
        }
        if (!line.empty()) {
            vec.mul(1.0 / line.size());
        }
        int vec_len = vec.m_;
        //float query[vec_len];
        //for (int i = 0; i < vec.m_; i++) {
            //query[i] = vec.data_[i];
            //a.push_back(vec.data_[i]);
       // }
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = appr_alg->searchKnn(vec.data_, k);
        std::cout << "result : \n";
        std::cout << "size is " << result.size() << "\n";

        while (!result.empty()) {
            std::cout << "query:" << content << std::endl;
            std::cout << "res : " << sents[result.top().second] << "\n" << result.top().first << std::endl;
            result.pop();
        }

    }

    void FastText::add_points(hnswlib::HierarchicalNSW<float>* appr_alg) {
        std::vector<int32_t> line, labels;
        Vector vec(args_->dim);
        std::ifstream in;
        in.open("/home/zxia/Desktop/stsutf8/test.txt");
        std::string content;
        int index = 0;
        while (std::getline(in, content)) {
            std::istringstream ss(content);
            dict_->getLine(ss, line, labels, model_->rng);
            dict_->addNgrams(line, args_->wordNgrams);
            vec.zero();
            for (auto it = line.cbegin(); it != line.cend(); ++it) {
                vec.addRow(*input_, *it);
            }
            if (!line.empty()) {
                vec.mul(1.0 / line.size());
            }
            int vec_len = vec.m_;
            //float sentence_array[vec_len];
            //for (int i = 0; i < vec.m_; i++) {
                //sentence_array[i] = vec.data_[i];
                //a.push_back(vec.data_[i]);
            //}
            appr_alg->addPoint(vec.data_, index);
            index++;
        }
    }


    hnswlib::HierarchicalNSW<float>* hnsw_init(int dim) {
        hnswlib::SpaceInterface<float>* l2space;
        hnswlib::SpaceInterface<float>* ipspace;
        l2space = new hnswlib::L2Space(dim);
        ipspace = new hnswlib::InnerProductSpace(dim);
        size_t M = 16;
        size_t ef_construction = 200;
        size_t max_element = 6000;
        hnswlib::HierarchicalNSW<float>* appr_alg = new hnswlib::HierarchicalNSW<float>(l2space, max_element, M, ef_construction, 100);
        return appr_alg;
    }

    std::vector<std::string> content_list(const std::string path) {
        std::vector<std::string> sents;
        std::ifstream in;
        in.open(path);
        std::string content;
        while (std::getline(in, content)) {
            sents.push_back(content);
        }
        return sents;
    }

    void FastText::test_nn() {
        int dim = args_->dim;
        hnswlib::HierarchicalNSW<float>* appr_alg = hnsw_init(dim);
        add_points(appr_alg);

        std::vector<std::string> sents = content_list("/home/zxia/Desktop/stsutf8/test.txt");
        for (int i = 0; i <= 500; i++) {
            find_knn_by_query(sents[i], appr_alg, 3, sents);
        }
    }

    void FastText::print_sentence_vectors(const std::string path, const std::string save_path)
    {
        std::vector<int32_t> line, labels;
        Vector vec(args_->dim);
		//std::ifstream in("D:\\test_sentences_ansi.txt");
		std::ifstream in(path);
        if (!in.is_open()) 
        {
            std::cout << "input file cannot be openned"<<"\n";
            exit(EXIT_FAILURE);
        }
        std::string content;
        int index = 0;
        //std::ofstream ofs("D:\\s2v_embeddings.txt");
		std::ofstream ofs(save_path);
        while (std::getline(in, content)) {
            content = content.substr(content.find_first_of(' ') + 1, content.size() - content.find_first_of(' '));
            std::istringstream ss(content);
            dict_->getLine(ss, line, labels, model_->rng);
            dict_->addNgrams(line, args_->wordNgrams);
            vec.zero();
            for (auto it = line.cbegin(); it != line.cend(); ++it) {
                vec.addRow(*input_, *it);
            }
            if (!line.empty()) {
                vec.mul(1.0 / line.size());
            }
            ofs << content << "\t";
            for (int i = 0; i < vec.m_; i++)
            {
				ofs << vec.data_[i] << " ";
            }
			ofs << std::endl;
        }
    }
}
