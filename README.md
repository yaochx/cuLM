cuLM
====

Fitting with Levenberg-Marquardt algorithm in CUDA.

It is based on the LM algorithm written by Joachim Wuttke and is available at his website http://joachimwuttke.de/ as `lmfit`.

This implementation does not optimize speed of a single fit! Purpose of this project is to perform fitting of multiple functions at once. That is exactly what I need to do in project called ThunderSTORM (http://code.google.com/p/thunder-storm/), where I perform fitting of thousands of molecules.
