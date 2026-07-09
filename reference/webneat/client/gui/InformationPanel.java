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

import javax.swing.*;
import java.text.DecimalFormat;

class InformationPanel extends JPanel {
	private final JTextField memory, threads, renderTime;

	DecimalFormat formatter = new DecimalFormat("#.###");
	
	private static InformationPanel instance = null;
	
	InformationPanel() {
		super();
		this.setLayout(new javax.swing.BoxLayout(this, javax.swing.BoxLayout.Y_AXIS));
		
		memory = createField("Memory");
		threads = createField("Threads");
		renderTime = createField("Last Render Time");
		
		new Updater().start();

		instance = this;	
	}
	
	static void setLastRenderTime(long ms) {
		if(instance != null)
			instance.renderTime.setText(instance.formatter.format(ms / 1000.0) + " secs.");
	}
	
	private JTextField createField(String name) {
		JTextField field = new JTextField("---");
		JLabel label = new JLabel(name + ": ");
		label.setLabelFor(field);
		
		add(label);
		add(field);
		
		field.setEditable(false);
		
		return field;
	}
	
	private class Updater extends Thread {
		public void run() {
			while(true) {
				update();
			}
		}
		
		private void update() {
			double mem = Runtime.getRuntime().totalMemory() / (1024.0 * 1024.0);
			
			memory.setText(formatter.format(mem) + " MB");
			threads.setText(Integer.toString(Thread.activeCount()));
			
			try {
				Thread.sleep(sleepTime);
			}
			catch(Exception e) {
			}
		}
		
		private static final int sleepTime = 1000;
	}
}
