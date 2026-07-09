/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.evolution.generators;

/**
 * Gotta try some of Chef's chocolate salty balls!
 * @author Nick

 */
public class SuperSweetGenerator extends GeneratorChooser {
	public SuperSweetGenerator() {
		addGenerator(new AddNodes(), 4);
		addGenerator(new CrazyNewLinkAdder(), 6);
		addGenerator(new MutateLinks(), 10);
		addGenerator(new MutateActivation(), 1);
		lock();
	}
}
