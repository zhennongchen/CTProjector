export GPU_CODE_FLAG= \
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_80,code=sm_80 \
-gencode=arch=compute_86,code=sm_86 \
-gencode=arch=compute_89,code=sm_89 \
-gencode=arch=compute_90,code=sm_90 \
-gencode=arch=compute_120,code=sm_120

SUBDIRS = prior projector

.PHONY: all cuda tensorflow ${SUBDIRS}

all: ${SUBDIRS}

${SUBDIRS}:
	$(MAKE) -C $@

cuda: $(addsuffix -cuda,${SUBDIRS})

$(addsuffix -cuda,${SUBDIRS}):
	$(MAKE) -C $(patsubst %-cuda,%,$@) cuda

tensorflow: $(addsuffix -tensorflow,${SUBDIRS})

$(addsuffix -tensorflow,${SUBDIRS}):
	$(MAKE) -C $(patsubst %-tensorflow,%,$@) tensorflow

$(addsuffix -clean,${SUBDIRS}):
	cd $(patsubst %-clean,%,$@); make clean

clean: $(addsuffix -clean,${SUBDIRS})