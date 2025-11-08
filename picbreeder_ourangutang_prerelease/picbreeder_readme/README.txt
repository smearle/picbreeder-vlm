README for WebNEAT client.
Prepared by Nick Beato

Organization:
Purpose
Packages
Dependencies
Evolution
Math
GUI

Building (For the website)

To build first, you must run

ant generatekey

This will generate the keystore with which you can sign the jar.  Then issue

ant jar

to compile, build and sign the jar.  Finally, issue

ant install

to copy the new jar file to the appropriate directory.

Purpose
The purpose of this document is to explain how the files are organized and
provide insight into my reasoning during design.

If you are working on the project, this is a good start to understand how the
software works.  Also, since I'm trying to rationalize things, this is the
place to look if you decide a feature is hard to add.  If this document states
the design was not supporting that feature, you might be in trouble.  Likewise,
if the document doesn't say anything about it, you might be in trouble.

Packages
There are a few main packages.  The are presented in the order that you
should understand them (conceptually).

client
The client package contains the client code.  Essentially, this entire package
contains the applet.  There are exceptions for "test" code... which really needs
to be moved by now.

client.evolution
The evolution package contains all information on genetic encodings and NEAT
code.  This is the package which lets you evolve CPPNs as abstract graphs.

client.cppn
The cppn package contains the CPPN activation code.  A CPPN is constructed from
evolved genomes.

client.evolution.generators
The generators are how evolution works.  A generator is a customizable, pluggable
architecture that let's you create a genome from a set of parents.  There are
a few reasons this implementation was chosen over the "normal" NEAT.  First off,
there are no species.  If you want them, sorry, you'll probably have to redo the
evolution.  Second, there are no fitness functions.  In other words, any
selected image has an uniform random chance of being used by a generator.  Third,
Adam's original applet allowed each row of images to spawn from different
operations.  The generators package is very robust for this.  Finally, a global
complex generator can be used to "simulate" the usual NEAT crossover/mutation
scheme.  I can think of many more reasons that an extendible architecture will
help WebNEAT.  So, again, if you need to include selection methods or speciation,
you are probably in bad shape.  Otherwise, you can probably implement a custom
generator and do what you want.

client.math
The math package contains the math parser.  Right now it is a bit "dirty", but
the main thing to understand is that it contains the methods to decode
the textual activation functions from the genomes to create the CPPNs. This
code is delegated to a package because we talked about evolving math equations
at some point.

client.utilities
The utilities package stores useful things like XML routines, the random generator,
compression routines, etc.  It is meant to be an independant package.

client.tools
The tools package stores command line programs that are useful outside of evolution.
For example, there is a routine to generate an image from a genome. There is also
a program that verifies that the database is consistent (as a bug checking mechanism).
There are also tools for extracting information from XML that the server might
want to know.

client.renderers
The renderers package contains various renderers.  Overwriting one of these should
let you apply dithering or multithreaded rendering.  Just remember that the output
layer of a network (by default) uses the same output array.  So you should process
an image/genome on no more than one thread simultaneously.

client.gui
The gui package is the hack that makes the applet and the application.  It works...
but it definately is not pretty.

client.server
The server contains the code to connect to the website. All communication is
done through the ServerConnectionInterface. The interface bloated as more
and more features were added, but hopefully it makes sense.

Dependencies
The dependency graph for almost everything is acyclic.  This is true at the package
level and at the class level.  Please, do NOT destroy this!  In general, I followed
"standard" C++ file/library/namespace design.

Parent dependencies
Given two nested packages, the subdirectory package may depend on its parent
packages, but never vice versa.  For example, the client.evolution package
depends on the client package, but not vice versa.  The only exception to
this rule is the default values for singletons (aka, the factories).  This dependency was
not nescessary, but it makes it nice that you don't have to set the singleton
for every executable class.

Acyclic nature
The package structure is forcably acyclic.  This is on purpose.  Do NOT
destroy this.  The reason for this (in ANY program) is 2-fold.  1) Anyone
who needs to learn the code can start at leaves of the dependancy tree and
easily work backwards to understand the project without having to understand
cross-dependant packages.  2) Any unit tests in a modified package will not be
able to propogate through the project in an unpredictable manner.  For example,
if someone modifies the gui package, and no packages depend on the GUI, then
no other packages can break.  However, if someone decides the cppn package
should depend on the GUI, then a change to the gui could break the CPPN package,
killing all of the tools.

IMPL packages
Any package marked impl is an implementation of its parent package.  It's scope
should make it practically invisible.

Dependencies (I hope someone updates this besides me)
Note: a -> b,c means that package a depends on package b and c (and so on)
client.evolution -> client
client.cppn -> client, evolution
client.evolution.generators -> client, client.evolution
TODO...

test
