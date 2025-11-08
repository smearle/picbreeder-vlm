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

package client.gui;

import javax.swing.JSlider;

class ExplorationSlider extends JSlider {
	ExplorationSlider() {
		super(0, 100, 50);
		
		final JSlider temp = this;
		
		this.addChangeListener(new javax.swing.event.ChangeListener() {
			public void stateChanged(javax.swing.event.ChangeEvent event) {
				double t = (double)(temp.getValue()) / (temp.getMaximum() - temp.getMinimum());
				client.evolution.GeneticFactoryInstance.get().setSliderCoeffecient(t);
			}
		});
		
		this.setValue((this.getMaximum() - this.getMinimum()) / 2);
	}
}
