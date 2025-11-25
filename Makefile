CC = clang
CFLAGS = -Wall -Werror -O3 -fPIC
LDFLAGS =

default: libcspg.so

clean:
	$(RM) *.o *.so

cspg.o: cspg.c cspg.h
	$(CC) -c -I. $(CFLAGS) $< -o $@

libcspg.so: cspg.o
	$(CC) $(LDFLAGS) -shared $< -o $@ -lm

.PHONY: default clean
