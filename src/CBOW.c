
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <limits.h>

#include <ctype.h>

#include "word2vec.h"

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

char input_file[MAX_STRING];
char output_file[MAX_STRING];
int binary = 1;
long long file_size = 0;

int n_of_thread;

// structures, vars for training
int *hash;
float* nodes;
int size_of_hash = MAX_VOCAB_SIZE;
struct WORD* vocab;
int size_of_vocab = 2048;
int n_of_words = 0;
int n_of_words_limit;
long long total_words = 0;
long long trained_word = 0;
int window_size = 5;
int min_count = 5;
int epoch;

float* expTable;
int n_of_inner_node = 0;
float starting_lr;
float lr;

float sample = 1e-5;
long long *skip_cnt;
long long total_skip_cnt = 0;

// model var
int hidden_size;
float* in_layer;

///////////////////////////////

void* training_thread(void* id_ptr){
    long long id = (long long)id_ptr;

    FILE* infp = fopen(input_file, "r");
    long long* sentence = (long long*)malloc(sizeof(long long)*MAX_SENTENCE_LENGTH);
    long long context_count;
    long long target, target_pos;
    long long context, context_pos;
    long long sentence_length;

    long long random_window;
    unsigned long long next_random = (long long)id;

    long long local_trained_word = 0; 
    long long local_last_trained_word = 0;
    float* layer_grad = (float*)calloc(hidden_size, sizeof(float));
    float* hidden_values = (float*)calloc(hidden_size, sizeof(float));

    long long word_per_thread = total_words/n_of_thread;
    long long local_skipped_total=0;

    lr = starting_lr;
    for(int ep=0; ep<epoch; ep++){
        clock_t start = time(NULL);

        fseek(infp, file_size / (long long)n_of_thread * (long long)id, SEEK_SET);
        local_trained_word = 0;
        local_last_trained_word = 0;
        if(id==0) printf("\nRunning epoch %d\n", ep);
        while(1){

            sentence_length = readSentenceFromFile(infp, sentence, id, ep+1);
            if(sentence_length < 0) break;
            local_trained_word += skip_cnt[id];
            local_skipped_total += skip_cnt[id];

            for(target_pos=0; target_pos<sentence_length; target_pos++){
                // traverse the sentence -> target
                // 0. Calculate current learning rate
                
                if (local_trained_word - local_last_trained_word > 10000){
                    trained_word += local_trained_word - local_last_trained_word;
                    local_last_trained_word = local_trained_word;
                    lr = starting_lr*(1-trained_word/(float)(epoch*total_words+1));
                    if(lr < starting_lr*0.0001) lr = starting_lr*0.0001;
                    if(id==0){
                        printf("\rLearning rate: %f, Progress: %.4f, current skipped words: %lld, time: %ld", lr, (float)(local_trained_word)/(float)(total_words/n_of_thread), local_skipped_total, time(NULL)-start);
                        fflush(stdout);
                    }
                }

                // 1. Set target
                target = sentence[target_pos];
                if(target==-1) continue;

                // 2. forward pass in_layer
                // reset values
                context_count = 0;
                for(int a=0; a<hidden_size; a++){
                    hidden_values[a] = 0.0;
                    layer_grad[a] = 0.0;
                }
                next_random = next_random * (unsigned long long)25214903917 + 11;
                random_window = next_random % window_size;
                for(context_pos=target_pos-random_window; context_pos<=target_pos+random_window; context_pos++){
                    if(context_pos < 0) continue;
                    if(context_pos >= sentence_length) break;

                    if(context_pos != target_pos){
                        context = sentence[context_pos];
                        if(context == -1) continue;
                        for(int b=0; b<hidden_size; b++){
                            hidden_values[b] += in_layer[context*hidden_size + b];
                        }
                        context_count++;
                    }
                }

                if(context_count==0) continue;
                for(int b=0; b<hidden_size; b++){ // average
                    hidden_values[b] /= context_count;
                }

                // forward pass out_layer
                // hierarchical softmax
                float g;
                for(int d=0; d<vocab[target].codelen; d++){
                    int current_path = vocab[target].point[d];

                    // dot product
                    float f = 0;
                    for(int b=0; b<hidden_size; b++){
                        f += hidden_values[b] * nodes[current_path*hidden_size + b];
                    }
                    //sigmoid
                    if(f<=-MAX_EXP || f>=MAX_EXP) continue;
                    else f = expTable[(int)((f+MAX_EXP)*(EXP_TABLE_SIZE/MAX_EXP/2))];

                    // 3. backward pass
                    g = (1 - vocab[target].code[d] - f);
                    g *= lr;

                    for(int b=0; b<hidden_size; b++){
                        // calculate gradient
                        layer_grad[b] += g * nodes[current_path*hidden_size + b];
                        // update inner node
                        nodes[current_path*hidden_size + b] += g*hidden_values[b];
                    }
                }

                // 4. update in_layer
                for(context_pos=target_pos-random_window; context_pos<=target_pos+random_window; context_pos++){
                    if(context_pos < 0) continue;
                    if(context_pos >= sentence_length) break;

                    if(context_pos != target_pos){
                        context = sentence[context_pos];
                        for(int b=0; b<hidden_size; b++){
                            in_layer[context*hidden_size + b] += layer_grad[b]; //check dividing with context_count
                        }
                    }
                }

                local_trained_word++;
            }

            if(local_trained_word > word_per_thread){
                trained_word += local_trained_word - local_last_trained_word;
                lr = starting_lr*(1-trained_word/(float)(epoch*total_words+1));
                if(lr < starting_lr*0.0001) lr = starting_lr*0.0001;
                if(id==0){
                    printf("\rLearning rate: %f, Progress: %.4f, current skipped words: %lld, time: %ld", lr, (float)(local_trained_word)/(float)(total_words/n_of_thread), local_skipped_total, time(NULL)-start);
                    fflush(stdout);
                }
                break;
            }
        }
    }

    free(hidden_values);
    free(layer_grad);
    free(sentence);
    fclose(infp);

    printf("Thread %lld returning\n", id);
    fflush(stdout);

    return NULL;
}

int main(int argc, char** argv){
    if(argc < 9){
        printf("Usage example: ./cbow hidden_size window_size n_of_words_limit sampling_param thread_number epoch data_file output_file\n");
        return -1;
    }
    else{
        hidden_size = atoi(argv[1]);
        window_size = atoi(argv[2]);
        n_of_words_limit = atoi(argv[3]);
        sample = atof(argv[4]);
        n_of_thread = atoi(argv[5]);
        epoch = atoi(argv[6]);
        strcpy(input_file, argv[7]);
        strcpy(output_file, argv[8]);
    }
    starting_lr = 0.05;
    printf("Starting learning rate : %f\n", starting_lr);
    printf("Sampling param: %f\n", sample);

    // prepare for training
    hash = (int*)calloc(size_of_hash, sizeof(int));
    vocab = (struct WORD*)calloc(size_of_vocab, sizeof(struct WORD));

    buildHash(input_file);
    buildBinaryTree();

    expTable = (float*)malloc((EXP_TABLE_SIZE + 1)*sizeof(float));
    for(int i=0; i<EXP_TABLE_SIZE; i++){
        expTable[i] = exp((i/(float)EXP_TABLE_SIZE*2 - 1)*MAX_EXP);
        expTable[i] = expTable[i] / (expTable[i] + 1);
    }

    // initialize model
    in_layer = (float*)malloc(sizeof(float)*hidden_size*n_of_words);
    int random_number = time(NULL);
    for(int a=0; a<n_of_words; a++){
        for(int b=0; b<hidden_size; b++){
            random_number = random_number * (unsigned long long)25214903917 + 11;
            in_layer[a*hidden_size + b] = (((random_number & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
        }
    } 

    // initialize inner nodes of binary tree
    printf("n_of_inner_node: %d\n",n_of_inner_node);

    nodes = (float*)malloc(sizeof(float)*hidden_size*n_of_inner_node);
    for(int a=0; a<n_of_inner_node; a++){
        for(int b=0; b<hidden_size; b++){
            random_number = random_number * (unsigned long long)25214903917 + 11;
            //nodes[a*hidden_size + b] = (((random_number & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
            nodes[a*hidden_size + b] = 0.0;
        }
    }

    // train
    printf("Training...");
    time_t start_time = time(NULL);
    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t)*n_of_thread);

    int* id = (int*)malloc(sizeof(int)*n_of_thread);
    skip_cnt = (long long*)malloc(sizeof(long long)*n_of_thread);
    for(int a=0; a<n_of_thread; a++){
        id[a] = a;
        pthread_create(&threads[a], NULL, training_thread, (void*)(long)a);
    }
    printf("all threads created\n");
    for(int a=0; a<n_of_thread; a++){
        pthread_join(threads[a], NULL);
    }
    time_t end_time = time(NULL);
    printf("\nTraining done... took %ld, last learning rate: %f trained_words: %lld skipped_words: %lld\n", end_time-start_time, lr, trained_word, total_skip_cnt);

    // save word vectors
    FILE* outfp = fopen(output_file, "wb");
    long long nonalphabet_cnt = 0;
    fprintf(outfp, "%lld %lld\n", (long long)n_of_words, (long long)hidden_size);
    for(int a=0; a<n_of_words; a++){
        fprintf(outfp, "%s ", vocab[a].word);

        if(binary) {
            for(int k=0;k<strlen(vocab[a].word);k++){
                if(!isalpha(vocab[a].word[k])){ nonalphabet_cnt++;}
            }

            for(int b=0; b<hidden_size; b++){
                fwrite(&in_layer[a*hidden_size + b], sizeof(float), 1, outfp);
            }
        }
        else{
            for(int b=0; b<hidden_size; b++){
                fprintf(outfp, "%lf ", in_layer[a*hidden_size+b]);
            }
        }
        fprintf(outfp, "\n");
    }
    fclose(outfp);
    // free everything
    free(id);
    free(hash);
    free(vocab);
    free(skip_cnt);
    free(threads);

    free(in_layer);
    
    printf("non-aphabet characters: %lld\n", nonalphabet_cnt);
    return 0;
}

void resetHashTable(){
    for(int i=0; i<size_of_hash; i++){
        hash[i] = -1;
    }
    return;
}

int getWordHash(char* word){
    int hash_key = 0;
    for(int i=0; i<strlen(word); i++){
        hash_key += (hash_key << 5)*i + word[i];
    }
    hash_key = hash_key % MAX_VOCAB_SIZE;
    return abs(hash_key);
}

void buildHash(char* file_name){
    printf("building hash table...\n");
    resetHashTable();

    FILE* infp = fopen(file_name, "r");
    printf("file_name %s\n", file_name);

    if(infp==NULL){ printf("file not found\n"); exit(1);}

    char ch;
    char* cur_word = (char*)calloc(MAX_STRING, sizeof(char));
    int word_length = 0;
    int hash_key;

    while((ch = fgetc(infp)) != EOF){
        if(ch==13) continue;
        if(ch == ' ' || ch == '\n' || ch == '\t' || ch == '\0'){
            if (word_length==0) continue;

            total_words++;

            cur_word[word_length] = 0;
            word_length = 0;
            hash_key = getWordHash(cur_word);

            while(1){
                if(hash[hash_key]==-1){
                    if(n_of_words >= size_of_vocab){
                        size_of_vocab += 2048;
                        vocab = realloc(vocab,size_of_vocab*sizeof(struct WORD));
                        if(vocab==NULL){
                            printf("Reallocation failed\n");
                            exit(1);
                        }
                    }
                    hash[hash_key] = n_of_words;
                    vocab[n_of_words].count = 1;
                    strcpy(vocab[n_of_words].word, cur_word);
                    n_of_words++;
                    break;
                }
                if(strcmp(vocab[hash[hash_key]].word, cur_word)==0){
                    vocab[hash[hash_key]].count++;
                    break;
                }
                hash_key = (hash_key + 1) % MAX_VOCAB_SIZE;
            }
        }
        else{
            if(word_length >= MAX_STRING - 1) word_length--;
            cur_word[word_length++] = ch;
        }
    }

    free(cur_word);
    file_size = ftell(infp);
    fclose(infp);
    printf("done... n_of_words = %d total_words = %lld\n", n_of_words, total_words);
    return;
}

void initModel(){
 // initialize model
}

int readSentenceFromFile(FILE* fp, long long* sentence, long long thread_id, int iter){
    char ch;
    char cur_word[MAX_STRING] = {0};
    int word_length = 0;
    int sentence_length = 0;
    int id_found;
    unsigned long long next_random = thread_id;
    next_random += (unsigned long long)iter*17;

    skip_cnt[thread_id] = 0;
    while(!feof(fp)){
        ch = fgetc(fp);
        if(ch==' ' || ch=='\t' || ch=='\n'){
            
            if(word_length==0) continue;
            cur_word[word_length] = 0;
            word_length = 0;

            id_found = searchVocabID(cur_word);
            if(id_found != -1){
                if (sample > 0){
                    float ran = (sqrt(vocab[id_found].count / (sample * total_words)) + 1) * (sample * total_words) / vocab[id_found].count;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if(ran < (next_random & 0xFFFF) / (float)65536) {
                        skip_cnt[thread_id]++;
                        total_skip_cnt++;
                        continue;
                    }
                }
                sentence[sentence_length++] = id_found;
            }

            if(ch=='\n') { return sentence_length;}
            if(sentence_length >= MAX_SENTENCE_LENGTH){
                return sentence_length;
            }
        }
        else if(ch=='\r') continue;
        else{
            if(word_length >= MAX_STRING - 1) word_length--;
            cur_word[word_length++] = ch;
        }
    }

    if(word_length > 0){
        // add the last word
        cur_word[word_length] = 0;
        word_length = 0;

        id_found = searchVocabID(cur_word);
        if(id_found != -1){
            if (sample > 0){
                float ran = (sqrt(vocab[id_found].count / (sample * total_words)) + 1) * (sample * total_words) / vocab[id_found].count;
                next_random = next_random * (unsigned long long)25214903917 + 11;
                if(ran < (next_random & 0xFFFF) / (float)65536) {
                    skip_cnt[thread_id]++;
                    total_skip_cnt++;
                    return sentence_length;
                }
            }
            sentence[sentence_length++] = id_found;
        }
    }
    if(sentence_length==0) return -1;
    return sentence_length;
}

int searchVocabID(char* word){
    int hash_key = getWordHash(word);
    while(1){
        if(hash[hash_key]==-1) return -1;
        if(strcmp(vocab[hash[hash_key]].word, word)==0) return hash[hash_key];
        hash_key = (hash_key+1) % MAX_VOCAB_SIZE;
    }
}

int _comp(const void* a, const void* b){
    return ((struct WORD*)b)->count - ((struct WORD*)a)->count;
}

void buildBinaryTree(){
    printf("building binary tree...\n");
    // 1. Sort vocab by count
    qsort(vocab, n_of_words, sizeof(struct WORD), _comp); // descending order

    // 2. Discard too less frequently appeared words
    // 3. Allocate space for codes
    // 4. recompute hash
    resetHashTable();
    total_words = 0;
    int hash_key;
    for(int i=0; i<n_of_words; i++){
        if (vocab[i].count < min_count || i >= n_of_words_limit) {
            n_of_words = i;
            break;
        }
        vocab[i].code = (char*)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[i].point = (int*)calloc(MAX_CODE_LENGTH, sizeof(int));

        hash_key = getWordHash(vocab[i].word);
        while(hash[hash_key]!=-1){
            hash_key = (hash_key + 1) % MAX_VOCAB_SIZE;
        }
        hash[hash_key] = i;

        total_words += vocab[i].count;
    }
    
    printf("n_of_words after excluding rare words %d\n", n_of_words);
    printf("total words after excluding rare words %lld\n", total_words);

    // 5. build binary tree
    int pos1, pos2, min1i, min2i;
    int* count = (int*)calloc(n_of_words*2 + 1, sizeof(int));
    int* binary = (int*)calloc(n_of_words*2 + 1, sizeof(int));
    int* parent_node = (int*)calloc(n_of_words*2 + 1, sizeof(int));

    for(int i=0; i<n_of_words; i++){
        count[i] = vocab[i].count;
    }
    for(int i=n_of_words; i<n_of_words*2; i++){
        count[i] = INT_MAX;
    }

    pos1 = n_of_words-1;
    pos2 = n_of_words;
    for(int i=0; i<n_of_words; i++){
        if(pos1 >= 0){ // find min1i
            if(count[pos1] < count[pos2]){
                min1i = pos1;
                pos1--;
            }
            else{
                min1i = pos2;
                pos2++;
            }
        }
        else{
            min1i = pos2;
            pos2++;
        }
        if(pos1 >= 0){ // find min2i
            if(count[pos1] < count[pos2]){
                min2i = pos1;
                pos1--;
            }
            else{
                min2i = pos2;
                pos2++;
            }
        }
        else{
            min2i = pos2;
            pos2++;
        }
        count[n_of_words+i] = count[min1i] + count[min2i];
        parent_node[min1i] = n_of_words+i;
        parent_node[min2i] = n_of_words+i;
        binary[min2i] = 1; // 1 for right node
    }
    // assign binary codes to each word
    printf("assigning codes...\n");
    int b, i;
    char* code = (char*)calloc(MAX_CODE_LENGTH, sizeof(char));
    int* point = (int*)calloc(MAX_CODE_LENGTH, sizeof(int));

    for(int a=0; a<n_of_words; a++){
        b = a;
        i = 0;
        while(1){ // find code of a by traversing from 'a' to root (by 'b')
            code[i] = binary[b];
            point[i] = b; // point = parent node
            i++;
            b = parent_node[b]; // follow parent node -> leads to root
            if(b==n_of_words*2-2){
                break;
            }
        }

        vocab[a].codelen = i;
        vocab[a].point[0] = n_of_words - 2;

        for( b=0; b<i; b++){
            vocab[a].code[i-b-1] = code[b]; // code is written backwards, so flip it!
            vocab[a].point[i-b] = point[b] - n_of_words; // storing parent nodes -> the path from root to a
            if(n_of_inner_node < point[b] - n_of_words) n_of_inner_node = point[b] - n_of_words;
        }

    }

    n_of_inner_node += 2;
    free(count);
    free(binary);
    free(parent_node);
    free(code);
    free(point);
    printf("done...\n");
    return;
}

char* IDtoWord(int id){
    return vocab[id].word;
}