# file: dll.mak
# description: Build VLFeat DLL
# author: Andrea Vedaldi

# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.
#
# This file is part of the VLFeat library and is made available under
# the terms of the BSD license (see the COPYING file).

all: dll-all
clean: dll-clean
archclean: dll-archclean
distclean: dll-distclean
info: dll-info

# --------------------------------------------------------------------
#                                                        Configuration
# --------------------------------------------------------------------

DLL_NAME = vl

DLL_CFLAGS  = $(STD_CFLAGS)
DLL_CFLAGS += -fvisibility=hidden -fPIC -DVL_BUILD_DLL -pthread
# Only add -msse2 for x86 architectures, not ARM64
ifneq ($(ARCH),maca64)
DLL_CFLAGS += $(call if-like,%_sse2,$*,-msse2)
endif

DLL_LDFLAGS += -lm

BINDIR = bin/$(ARCH)

# Mac OS X on Intel 32 bit processor
ifeq ($(ARCH),maci)
DLL_SUFFIX := dylib
endif

# Mac OS X on Intel 64 bit processor
ifeq ($(ARCH),maci64)
DLL_SUFFIX := dylib
endif

# Mac OS X on ARM64 (Apple Silicon) processor
ifeq ($(ARCH),maca64)
DLL_SUFFIX := dylib
endif

# Linux-32
ifeq ($(ARCH),glnx86)
DLL_SUFFIX := so
endif

# Linux-64
ifeq ($(ARCH),glnxa64)
DLL_SUFFIX := so
endif

# --------------------------------------------------------------------
#                                                                Build
# --------------------------------------------------------------------

# On Mac OS X the library install_name is prefixed with @loader_path/.
# At run time this causes the loader to search for a local copy of the
# library for any binary which is linked against it. The install_name
# can be modified later by install_name_tool.

dll_tgt := $(BINDIR)/lib$(DLL_NAME).$(DLL_SUFFIX)
dll_src := $(wildcard $(VLDIR)/vl/*.c)
# Exclude SSE2 files for ARM64 architecture
ifeq ($(ARCH),maca64)
dll_src := $(filter-out %_sse2.c, $(dll_src))
endif
dll_hdr := $(wildcard $(VLDIR)/vl/*.h)
dll_obj := $(addprefix $(BINDIR)/objs/, $(notdir $(dll_src:.c=.o)))
dll_dep := $(dll_obj:.o=.d)

arch_bins += $(dll_tgt)
comm_bins +=
deps += $(dll_dep)

.PHONY: dll
.PHONY: dll-all, dll-clean, dll-archclean, dll-distclean
.PHONY: dll-info
no_dep_targets += dll-dir dll-clean dll-archclean dll-distclean
no_dep_targets += dll-info

dll-all: dll
dll: $(dll_tgt)

# generate the dll-dir target
$(eval $(call gendir, dll, $(BINDIR) $(BINDIR)/objs))

$(BINDIR)/objs/%.o : $(VLDIR)/vl/%.c $(dll-dir)
	$(call C,CC) $(DLL_CFLAGS)                                   \
	       -c "$(<)" -o "$(@)"

$(BINDIR)/objs/%.d : $(VLDIR)/vl/%.c $(dll-dir)
	$(call C,CC) $(DLL_CFLAGS)                                   \
	       -M -MT '$(BINDIR)/objs/$*.o $(BINDIR)/objs/$*.d'      \
	       "$(<)" -MF "$(@)"

# For ARM64 (maca64), use clang directly instead of libtool
ifeq ($(ARCH),maca64)
$(BINDIR)/lib$(DLL_NAME).dylib : $(dll_obj)
	$(call C,CC) -dynamiclib                                    \
                    -install_name @loader_path/lib$(DLL_NAME).dylib  \
	            -compatibility_version $(VER)                    \
                    -current_version $(VER)                          \
                    -o $@ $^                                        \
	            $(DLL_LDFLAGS)
else
$(BINDIR)/lib$(DLL_NAME).dylib : $(dll_obj)
	$(call C,LIBTOOL) -dynamic                                   \
                    -flat_namespace                                  \
                    -install_name @loader_path/lib$(DLL_NAME).dylib  \
	            -compatibility_version $(VER)                    \
                    -current_version $(VER)                          \
                    -syslibroot $(SDKROOT)                           \
		    -macosx_version_min $(MACOSX_DEPLOYMENT_TARGET)  \
	            -o $@ -undefined suppress $^                     \
	            $(DLL_LDFLAGS)
endif


$(BINDIR)/lib$(DLL_NAME).so : $(dll_obj)
	$(call C,CC) $(DLL_CFLAGS) -shared $(^) -o $(@) $(DLL_LDFLAGS)

dll-clean:
	rm -f $(dll_dep) $(dll_obj)

dll-archclean: dll-clean
	rm -rf $(BINDIR)

dll-distclean:
	rm -rf bin

dll-info:
	$(call echo-title,VLFeat core library)
	$(call dump-var,dll_hdr)
	$(call dump-var,dll_src)
	$(call dump-var,dll_obj)
	$(call dump-var,dll_dep)
	$(call echo-var,BINDIR)
	$(call echo-var,DLL_NAME)
	$(call echo-var,DLL_CFLAGS)
	$(call echo-var,DLL_LDFLAGS)
	$(call echo-var,DLL_SUFFIX)
	$(call echo-var,LIBTOOL)
	@echo

# Local variables:
# mode: Makefile
# End:
